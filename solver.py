import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

            #TODO

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST,
                                    data_dir=self.args.data_dir)
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))

    def fit(self):
        #TODO
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

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
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
        torch.save(self.net.state_dict(), save_path)

    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
