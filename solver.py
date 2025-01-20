import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime  # per verificare i tempi di addestramento
import torchvision.transforms as transforms

from model import Discriminator, GlobalNet, RefinementNet
from dataset import DepthDataset
from utils import visualize_img, ssim, plot_metrics

class Solver:

    def __init__(self, args):
        # prepare a dataset
        self.args = args

        # Definisco le trasformazioni per la data augmentation
        data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((144, 256), scale=(0.9, 1.0))
        ])

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # modelli
        self.globalnet = GlobalNet().to(self.device)
        self.refnet = RefinementNet().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=data_augmentation)
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

            # ottimizzatori
            self.optimizer_G = torch.optim.Adam(self.globalnet.parameters(), lr=args.lr)
            self.optimizer_R = torch.optim.Adam(self.refnet.parameters(), lr=args.lr)
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)

            # pesi delle loss
            self.lambda_adv = 100

            # train/val
            self.set = DepthDataset.TRAIN
            #self.set = DepthDataset.VAL

            # salvataggio delle metriche addestramento global
            self.gl_train_ssim_epochs = []
            self.gl_train_rmse_epochs = []

            # salvataggio delle metriche  addestramento adversarial
            self.adv_train_ssim_epochs = []
            self.adv_train_rmse_epochs = []

            self.args = args

            if not os.path.exists(args.gl_ckpt_dir):
                os.makedirs(args.gl_ckpt_dir)
            if not os.path.exists(args.ref_ckpt_dir):
                os.makedirs(args.ref_ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST,  # train = DepthDataset.VAL per il val set
                                         data_dir=self.args.data_dir)  # risistemare prima del commit
            gl_ckpt_file = os.path.join("checkpoint", "global", self.args.gl_ckpt_file)
            ref_ckpt_file = os.path.join("checkpoint", "refinement", self.args.ref_ckpt_file)
            self.globalnet.load_state_dict(torch.load(gl_ckpt_file, weights_only=True))
            self.refnet.load_state_dict(torch.load(ref_ckpt_file, weights_only=True))

    def globalnet_fit(self):
        print("GLOBAL FITTING")
        for epoch in range(self.args.max_epochs):
            self.globalnet.train()

            for images, depth in self.train_loader:
                images, depth = images.to(self.device), depth.to(self.device)

                fake_depth = self.globalnet(images)
                g_loss = self.criterion_rec(fake_depth, depth)

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

            # monitoraggio addestramento
            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/{}] Loss Global Net: {:.3f} Time:{}".format(epoch + 1, self.args.max_epochs, g_loss,
                                                                         curr_clock))
            epoch += 1

            if epoch % 10 == 0:
                self.global_evaluate()

            # salvataggio del checkpoint
            if epoch == self.args.max_epochs:
                self.save_global(self.args.gl_ckpt_dir, self.args.gl_ckpt_name, epoch)
                print("GLOBAL FITTING TERMINATO")

    def refnet_pretrain(self):
        print("\n\nREFINEMENT NET PRE-TRAINING")
        # per evitare di riaddestrare ogmi volta la global
        gl_ckpt_file = os.path.join("checkpoint", "global", self.args.gl_ckpt_file)
        self.globalnet.load_state_dict(torch.load(gl_ckpt_file, weights_only=True))

        for epoch in range(10):
            self.refnet.train()

            for images, depth in self.train_loader:
                images, depth = images.to(self.device), depth.to(self.device)

                fake_depth = self.refnet(self.globalnet(images), images)
                ref_loss = self.criterion_rec(fake_depth, depth)

                # addestramento refinement
                self.optimizer_R.zero_grad()
                ref_loss.backward()
                self.optimizer_R.step()

            # monitoraggio addestramento
            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/10] Loss R: {:.3f} Time:{}".format(epoch + 1, ref_loss, curr_clock))
            epoch += 1

            # salvataggio del checkpoint
            if epoch == 10:
                self.adv_evaluate() # metriche prima dell'addestramento avversario

    def adversarial_fit(self):
        print("\nADVERSARIAL FITTING")
        for epoch in range(self.args.max_epochs):
            self.refnet.train()
            self.discriminator.train()
            total_loss = 0
            lambda_adv = 100  # valore iniziale
            alpha = 0.05  # velocità di adattamento

            for images, depth in self.train_loader:
                images, depth = images.to(self.device), depth.to(self.device)

                # addestramento discriminatore
                self.optimizer_D.zero_grad()
                fake_depth = self.refnet(self.globalnet(images), images)

                real_labels = torch.ones((images.size(0), 1)).to(self.device)
                fake_labels = torch.zeros((images.size(0), 1)).to(self.device)

                real_loss = self.criterion_adv(self.discriminator(images, depth), real_labels)
                fake_loss = self.criterion_adv(self.discriminator(images, fake_depth.detach()), fake_labels)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # addestramento refinement
                self.optimizer_R.zero_grad()
                adv_loss = self.criterion_adv(self.discriminator(images, fake_depth), real_labels)
                rec_loss = self.criterion_rec(fake_depth, depth)

                # aggiornamento lambda_rec
                #ratio = rec_loss.item() / (adv_loss.item()+ 1e-8)
                #lambda_adv *= ratio ** alpha
                #lambda_adv = max(0.01, min(lambda_adv, 100.0))

                # calcolo della loss
                r_loss = rec_loss + self.lambda_adv * adv_loss
                total_loss += r_loss.item()
                r_loss.backward()
                self.optimizer_R.step()

            # monitoraggio addestramento
            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/{}] Loss R: {:.3f} Loss D: {:.3f} Time:{}".format(epoch + 1, self.args.max_epochs,
                                                                               r_loss, d_loss, curr_clock))
            epoch += 1

            if epoch % 10 == 0:
                self.adv_evaluate()

            if epoch == self.args.max_epochs:
                self.save_refinenment(self.args.ref_ckpt_dir, self.args.ref_ckpt_name, epoch)
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

        # salvataggio metriche per il grafico
        self.gl_train_ssim_epochs.append(ssim_acc / len(loader))
        self.gl_train_rmse_epochs.append(rmse_acc / len(loader))

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
        # checkpoint global
        save_path_gl = os.path.join(gl_ckpt_dir, "{}_{}.pth".format(gl_ckpt_name, global_step))
        torch.save(self.globalnet.state_dict(), save_path_gl)
        print("Checkpoint salvato")

    def save_refinenment(self, ref_ckpt_dir,ref_ckpt_name, global_step):
        # checkpoint refinement
        save_path_ref = os.path.join(ref_ckpt_dir, "{}_{}.pth".format(ref_ckpt_name, global_step))
        torch.save(self.refnet.state_dict(), save_path_ref)
        print("Checkpoint salvato")

    def test(self):
        loader = DataLoader(self.test_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.refnet(self.globalnet(images.to(self.device)), images.to(self.device))
                #output = self.globalnet(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(), depth[0].cpu(), output[0].cpu().detach(), suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))


