import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import Generator, Discriminator, UNet

from dataset import DepthDataset
from utils import visualize_img, ssim


class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args

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

            # turn on the CUDA if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # modelli
            #self.generator = Generator().to(self.device)
            self.discriminator = Discriminator().to(self.device)
            self.UNet = UNet().to(self.device) # prova U-Net TODO sistemare dopo U-Net

            #criteri
            self.criterion_adv = torch.nn.BCELoss()
            self.criterion_rec = torch.nn.L1Loss()

            # ottimizzatori
            #self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr) # TODO sistemare dopo U-Net
            self.optimizer_G = torch.optim.Adam(self.UNet.parameters(), lr=args.lr)
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)

            self.args = args

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST, # train = DepthDataset.VAL per il val set
                                    data_dir=self.args.data_dir)  # risistemare prima del commit
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            # self.generator.load_state_dict(torch.load(ckpt_file, weights_only=True)) # TODO sistemare dopo U-Net
            self.UNet.load_state_dict(torch.load(ckpt_file, weights_only=True))

    def fit(self):

        for epoch in range(self.args.max_epochs):
            #self.generator.train() # TODO sistemare dopo U-Net
            self.UNet.train()
            self.discriminator.train()
            for images, depth in self.train_loader:

                images = images.to(self.device)
                depth = depth.to(self.device)

                print(f"Input device: {images.device}, Model device: {next(self.UNet.parameters()).device}")

                # Train Discriminator
                self.optimizer_D.zero_grad()
                #fake_depth = self.generator(images) # TODO sistemare dopo U-Net
                fake_depth = self.UNet(images)

                real_labels = torch.ones((images.size(0), 1)).to(self.device)
                fake_labels = torch.zeros((images.size(0), 1)).to(self.device)

                real_loss = self.criterion_adv(self.discriminator(images, depth), real_labels)
                fake_loss = self.criterion_adv(self.discriminator(images, fake_depth.detach()), fake_labels)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                adv_loss = self.criterion_adv(self.discriminator(images, fake_depth), real_labels)

                rec_loss = self.criterion_rec(fake_depth, depth)
                g_loss = adv_loss + self.args.lambda_rec * rec_loss
                g_loss.backward()
                self.optimizer_G.step()
        return


    def evaluate(self, set):

        args = self.args
        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        #self.generator.eval()
        self.UNet.eval()
        # TODO sistemare dopo U-Net
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                #output = self.generator(images.to(self.device)) # TODO sistemare dopo U-Net
                output = self.UNet(images.to(self.device))

                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save({
            # 'generator_state_dict': self.generator.state_dict(), TODO sistemare dopo U-Net
            'generator_state_dict': self.UNet.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }, save_path)

    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                #output = self.generator(images.to(self.device)) TODO sistemare dopo U-Net
                output = self.UNet(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
