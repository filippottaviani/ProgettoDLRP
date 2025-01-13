import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime # per verificare i tempi di addestramento

from model import Discriminator, Generator
import torchvision.transforms as transforms
from dataset import DepthDataset
from utils import visualize_img, ssim


class Solver():

    def __init__(self, args):
        self.args = args

        if self.args.is_train:

            # Definisco le trasformazioni per la data augmentation
            data_augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((144, 256), scale=(0.9, 1.0))
            ])

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

            # turn on the CUDA if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # modelli
            self.generator = Generator().to(self.device)
            self.discriminator = Discriminator().to(self.device)

            #criteri
            self.criterion_adv = torch.nn.BCELoss()
            self.criterion_rec = torch.nn.L1Loss()

            # ottimizzatori
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr)
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)

            # pesi delle loss
            self.lambda_rec = 7
            self.lambda_adv = 1

            # train/val
            self.set = DepthDataset.TRAIN
            #self.set = DepthDataset.VAL

            # salvataggio delle metriche
            self.train_ssim_epochs = []
            self.train_rmse_epochs = []

            self.args = args

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.Generator = Generator().to(self.device)
            self.test_set = DepthDataset(train=DepthDataset.TEST,  # train = DepthDataset.VAL per il val set
                                         data_dir=self.args.data_dir)  # risistemare prima del commit
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.Generator.load_state_dict(torch.load(ckpt_file, weights_only=True))

    def fit(self):
        for epoch in range(self.args.max_epochs):
            self.generator.train()
            self.discriminator.train()

            for images, depth in self.train_loader:
                images = images.to(self.device)
                depth = depth.to(self.device)

                # addestramento discriminatore
                self.optimizer_D.zero_grad()
                fake_depth = self.generator(images)

                real_labels = torch.ones((images.size(0), 1)).to(self.device)
                fake_labels = torch.zeros((images.size(0), 1)).to(self.device)

                real_loss = self.criterion_adv(self.discriminator(images, depth), real_labels)
                fake_loss = self.criterion_adv(self.discriminator(images, fake_depth.detach()), fake_labels)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # addestramento generatore
                self.optimizer_G.zero_grad()
                adv_loss = self.criterion_adv(self.discriminator(images, fake_depth), real_labels)
                rec_loss = self.criterion_rec(fake_depth, depth)
                #rec_loss = ssim(fake_depth,depth) # prova

                g_loss = self.lambda_adv * adv_loss + self.lambda_rec * rec_loss
                g_loss.backward()
                self.optimizer_G.step()

            time = datetime.datetime.now()
            curr_clock = time.strftime("%H:%M:%S")
            print("Epoch [{}/{}] Loss G: {:.3f} Loss D: {:.3f} Time:{}".format(epoch + 1, self.args.max_epochs,
                                                                                g_loss, d_loss, curr_clock))
            epoch +=1

            if epoch % 10 == 0:
                self.evaluate()

            if epoch == self.args.max_epochs:
                self.save(self.args.ckpt_dir, "depth_final.pth", epoch)
                self.plot_metrics()
        return

    def evaluate(self):
        args = self.args
        if self.set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"

        elif self.set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"

        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

        self.generator.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.generator(images.to(self.device))

                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(), depth[0].cpu(), output[0].cpu().detach(), suffix=suffix)

        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader),"\n")

        # salvataggio metriche per il grafico
        self.train_ssim_epochs.append(ssim_acc / len(loader))
        self.train_rmse_epochs.append(rmse_acc / len(loader))

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save({'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }, save_path)
        print("Checkpoint salvato")

    def test(self):
        loader = DataLoader(self.test_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.Generator(images.to(self.device))

                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(), depth[0].cpu(), output[0].cpu().detach(), suffix="TEST")

        print("RMSE on TEST:", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))

    def plot_metrics(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_ssim_epochs, label="SSIM")
        plt.plot(self.train_rmse_epochs, label="RMSE")
        plt.xlabel("Epoca")
        plt.ylabel("Valore Metriche")
        plt.title("Andamento SSIM e RMSE durante il Training")
        plt.legend()
        plt.grid(True)
        plt.show()
