import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from math import exp
from matplotlib import pyplot as plt

def visualize_img(img_tensor, depth_tensor, pred_tensor, suffix):
    img = img_tensor.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose
    gt = depth_tensor.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose
    pred = pred_tensor.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose
    # Create a figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title('Input')
    axs[1].imshow(gt, cmap='magma')
    axs[1].set_title(f'True')
    axs[2].imshow(pred, cmap='magma')
    axs[2].set_title(f'Predicted')
    fig.suptitle(suffix)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(20)
    plt.close()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def random_crop_pair(self, image, depth):
    _, h, w = image.shape
    top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
    left = torch.randint(0, w - self.crop_size + 1, (1,)).item()

    cropped_image = TF.crop(image, top, left, self.crop_size, self.crop_size)
    cropped_depth = TF.crop(depth, top, left, self.crop_size, self.crop_size)

    return cropped_image, cropped_depth

def plot_metrics(self):
    plt.plot(self.adv_train_rmse_epochs, label="ADV RMSE")
    plt.xlabel("Intervalli di valutazione (ogni 10 epoche)")
    plt.ylabel("RMSE")
    plt.title("Andamento RMSE durante gli addestramenti")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(20)
    plt.savefig('rmse_metrics.png')
    plt.close()

    plt.plot(self.adv_train_ssim_epochs, label="ADV SSIM")
    plt.xlabel("Intervalli di valutazione (ogni 10 epoche)")
    plt.ylabel("SSIM")
    plt.title("Andamento SSIM durante gli addestramenti")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(20)
    plt.savefig('ssim_metrics.png')
    plt.close()