import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime  # per verificare i tempi di addestramento
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Discriminator, GlobalNet, RefinementNet
from dataset import DepthDataset
from utils import visualize_img, ssim, plot_metrics, random_crop_pair

class Solver:

    def __init__(self, args):
        # prepare a dataset
        self.args = args

        # impostazioni del training
        self.args.is_train = True
        self.args.only_global = False

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # modelli
        self.globalnet = GlobalNet().to(self.device)
        self.refnet = RefinementNet().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=None)
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir,
                                         transform=None)
            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=args.batch_size,
                                           num_workers=4,
                                           shuffle=True, drop_last=True)

            # criteri
            self.criterion_adv = torch.nn.BCELoss()
            self.criterion_rec = torch.nn.L1Loss()
            self.criterion_rec1 = torch.nn.MSELoss() # prova

            # ottimizzatori
            self.optimizer_G = torch.optim.Adam(self.globalnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
            self.optimizer_R = torch.optim.SGD(self.refnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
            self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

            # scheduling di riduzione della LR
            self.scheduler_R = ReduceLROnPlateau(self.optimizer_R, mode='min', factor=0.5, patience=5)
            self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5, patience=5)

            # pesi delle loss
            self.lambda_start = 5

            # fattore di crop per la refinement
            self.crop_size = 128

            # train/val
            self.set = DepthDataset.TRAIN
            #self.set = DepthDataset.VAL

            # salvataggio delle loss durante gli addestramenti
            self.glb_train_loss = []
            self.pre_train_loss = []
            self.adv_r_train_loss = []
            self.adv_d_train_loss = []

            # salvataggio delle metriche addestramento adversarial
            self.adv_train_ssim_epochs = []
            self.adv_train_rmse_epochs = []

            self.args = args

            if not os.path.exists(args.gl_ckpt_dir):
                os.makedirs(args.gl_ckpt_dir)
            if not os.path.exists(args.ref_ckpt_dir):
                os.makedirs(args.ref_ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.VAL,# train = DepthDataset.VAL per il val set
                                         data_dir=self.args.data_dir)  # risistemare prima del commit
            gl_ckpt_file = os.path.join("checkpoint", "global", self.args.gl_ckpt_file)
            ref_ckpt_file = os.path.join("checkpoint", "refinement", self.args.ref_ckpt_file)
            self.globalnet.load_state_dict(torch.load(gl_ckpt_file, weights_only=True))
            self.refnet.load_state_dict(torch.load(ref_ckpt_file, weights_only=True))

    def globalnet_fit(self):
        print("GLOBAL FITTING")
        for epoch in range(self.args.max_epochs):
            self.globalnet.train()
            total_g_loss = 0

            for images, depth in self.train_loader:
                images, depth = images.to(self.device), depth.to(self.device)

                fake_depth = self.globalnet(images)
                g_loss = self.criterion_rec(fake_depth, depth)

                self.optimizer_G.zero_grad()
                total_g_loss += g_loss.item()
                g_loss.backward()
                self.optimizer_G.step()

            # aggiornamento della loss
            avg_g_loss = total_g_loss / len(self.train_loader)
            self.scheduler_G.step(avg_g_loss)

            # monitoraggio addestramento
            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/{}] Loss Global Net: {:.3f} Time:{}".format(epoch + 1, self.args.max_epochs, g_loss,
                                                                         curr_clock))
            epoch += 1

            # per valutazione andamento della loss
            self.glb_train_loss.append(g_loss)

            if epoch % 10 == 0:
                self.global_evaluate()

            # salvataggio del checkpoint
            if epoch == self.args.max_epochs:
                self.save_global(self.args.gl_ckpt_dir, self.args.gl_ckpt_name, epoch)

    def refnet_pretrain(self):
        print("\n\nREFINEMENT NET PRE-TRAINING")
        gl_ckpt_file = os.path.join("checkpoint", "global", self.args.gl_ckpt_file)
        self.globalnet.load_state_dict(torch.load(gl_ckpt_file, weights_only=True))  # per evitare di riaddestrare ogmi volta la global
        total_r_loss = 0

        for epoch in range(10):
            self.refnet.train()

            for images, depth in self.train_loader:
                images, depth = images.to(self.device), depth.to(self.device)

                fake_depth = self.refnet(self.globalnet(images), images)
                ref_loss = self.criterion_rec(fake_depth, depth)

                # addestramento refinement
                self.optimizer_R.zero_grad()
                ref_loss.backward()
                total_r_loss += ref_loss.item()
                self.optimizer_R.step()

            # aggiornamento della loss
            avg_r_loss = total_r_loss / len(self.train_loader)
            self.scheduler_R.step(avg_r_loss)

            # monitoraggio addestramento
            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/10] Loss R: {:.3f} Time:{}".format(epoch + 1, ref_loss, curr_clock))
            epoch += 1

            # per valutazione andamento della loss
            self.pre_train_loss.append(ref_loss)

            # salvataggio del checkpoint
            if epoch == 10:
                self.adv_evaluate()  # metriche prima dell'addestramento avversario
                self.save_refinenment_final(self.args.ref_ckpt_dir, self.args.ref_ckpt_name, epoch)

    def adversarial_fit(self):
        print("\nADVERSARIAL FITTING")
        ref_ckpt_file = os.path.join("checkpoint", "refinement", "ref_depth_10.pth")
        self.refnet.load_state_dict(
            torch.load(ref_ckpt_file, weights_only=True))  # per evitare di preaddestrare ogni volta la refinement
        for epoch in range(self.args.max_epochs):
            self.refnet.train()
            self.discriminator.train()
            total_r_loss = 0
            lambda_adv = self.lambda_start  # valore iniziale
            alpha = 0.05  # velocità di adattamento

            for images, depth in self.train_loader:
                images, depth = images.to(self.device), depth.to(self.device)

                # Genera crop per ciascuna immagine nel batch
                cropped_images, cropped_depths = [], []
                for img, dpt in zip(images, depth):
                    crop_img, crop_dpt = random_crop_pair(self, img, dpt)
                    cropped_images.append(crop_img)
                    cropped_depths.append(crop_dpt)

                # Converte in batch tensor
                cr_images = torch.stack(cropped_images)
                cr_depths = torch.stack(cropped_depths)

                # addestramento discriminatore
                self.optimizer_D.zero_grad()
                fake_depth = self.refnet(self.globalnet(cr_images), cr_images)

                real_labels = torch.ones((cr_images.size(0), 1)).to(self.device)
                fake_labels = torch.zeros((cr_images.size(0), 1)).to(self.device)

                real_loss = self.criterion_adv(self.discriminator(cr_images, cr_depths), real_labels)
                fake_loss = self.criterion_adv(self.discriminator(cr_images, fake_depth.detach()), fake_labels)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # addestramento refinement
                self.optimizer_R.zero_grad()
                adv_loss = self.criterion_adv(self.discriminator(cr_images, fake_depth), real_labels)
                rec_loss = self.criterion_rec(fake_depth, cr_depths)
                ref_loss1 = self.criterion_rec1(fake_depth, cr_depths)
                ref_loss2 = ssim(fake_depth, cr_depths)

                # aggiornamento lambda_rec
                #ratio = rec_loss.item() / (adv_loss.item()+ 1e-8)
                #lambda_adv *= ratio ** alpha
                #lambda_adv = max(0.01, min(lambda_adv, 5.0))

                # calcolo della loss
                r_loss = rec_loss + lambda_adv * adv_loss + ref_loss1 + 1/ref_loss2
                total_r_loss += r_loss.item()
                r_loss.backward()
                self.optimizer_R.step()

            # aggiornamento della loss
            avg_r_loss = total_r_loss / len(self.train_loader)
            self.scheduler_R.step(avg_r_loss)

            # monitoraggio addestramento
            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/{}] Loss R: {:.3f} Loss D: {:.3f} Refinement LR:{} Time:{}".format(epoch + 1,
                                                                                                self.args.max_epochs,
                                                                                                r_loss, d_loss,
                                                                                                self.optimizer_R.param_groups[0]['lr'],
                                                                                                curr_clock))
            epoch += 1

            # per valutazione andamento della loss
            self.adv_r_train_loss.append(r_loss)
            self.adv_d_train_loss.append(d_loss)

            if epoch % 10 == 0:
                self.adv_evaluate()

            if epoch == self.args.max_epochs:
                self.save_refinenment_final(self.args.ref_ckpt_dir, self.args.ref_ckpt_name, epoch)
                plot_metrics(self)

    def global_evaluate(self):
        args = self.args
        if self.set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "GLOBAL TRAIN"
        elif self.set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

        self.globalnet.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.globalnet(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader),"\n")

    def adv_evaluate(self):
        args = self.args
        if self.set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "ADVERSARIAL TRAIN"
        elif self.set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

        self.refnet.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.refnet(self.globalnet(images.to(self.device)), images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                #if i % self.args.visualize_every == 0:
                    #visualize_img(images[0].cpu(), depth[0].cpu(),output[0].cpu().detach(), suffix=suffix)
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader),"\n")

        # salvataggio metriche per il grafico
        self.adv_train_ssim_epochs.append(ssim_acc / len(loader))
        self.adv_train_rmse_epochs.append(rmse_acc / len(loader))

    def save_global(self, gl_ckpt_dir,  gl_ckpt_name, global_step):
        save_path_gl = os.path.join(gl_ckpt_dir, "{}_{}.pth".format(gl_ckpt_name, global_step))
        torch.save(self.globalnet.state_dict(), save_path_gl) # checkpoint global
        print("Checkpoint salvato")

    def save_refinenment_pre(self, ref_ckpt_dir,ref_ckpt_name, global_step):
        save_path_ref = os.path.join(ref_ckpt_dir, "{}_{}.pth".format(ref_ckpt_name, global_step))
        torch.save(self.refnet.state_dict(), save_path_ref) # checkpoint refinement
        print("Checkpoint salvato")

    def save_refinenment_final(self, ref_ckpt_dir, ref_ckpt_name, global_step):
        save_path_ref = os.path.join(ref_ckpt_dir, "{}_{}.pth".format(ref_ckpt_name, global_step))
        torch.save(self.refnet.state_dict(), save_path_ref)  # checkpoint refinement
        print("Checkpoint salvato")

    def test(self):
        loader = DataLoader(self.test_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False,
                            drop_last=False)
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                if self.args.only_global:
                    output = self.globalnet(images.to(self.device))
                else:
                    output = self.refnet(self.globalnet(images.to(self.device)), images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(), depth[0].cpu(), output[0].cpu().detach(), suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))


