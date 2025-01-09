# models.py
import torch
import torch.nn as nn
from torch.xpu import device


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), nn.Tanh()
        )

        # refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Mappa finale di profondità tra -1 e 1
        )

    def forward(self, x):
        # global
        out_enc = self.encoder(x)
        out_dec = self.decoder(out_enc)

        # concatenazione
        combined_input = torch.cat((x, out_dec), dim=1)

        # refinement
        ref_output = self.refinement(combined_input)

        return ref_output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # elaborazione RGB
        self.rgb_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        # elaborazione Depth
        self.depth_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        # concatenazione e ulteriori convoluzioni
        self.joint_path = nn.Sequential(

            # conv1
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), nn.BatchNorm2d(64),

            # conv2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), nn.BatchNorm2d(128),

            # conv3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), nn.BatchNorm2d(256),

            # conv4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), nn.BatchNorm2d(512)
        )

        # layer completamente connessi
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1),  # output binario (vero/falso)
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_path(rgb)
        depth_features = self.depth_path(depth)

        # concatenazione lungo il canale
        joint_features = torch.cat((rgb_features, depth_features), dim=1)
        joint_output = self.joint_path(joint_features)

        # passaggio al livello completamente connesso
        output = self.fc(joint_output)
        return output


#   U-NET
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Blocchi di encoding
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Bottleneck
        self.bottleneck = self._block(512, 1024)

        # Blocchi di decoding
        self.dec4 = self._block(1024, 512)
        self.dec3 = self._block(512, 256)
        self.dec2 = self._block(256 , 128)
        self.dec1 = self._block(128, 64)

        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling e upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.up = nn.ConvTranspose2d non viene caricato correttamente in GPU
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2).to(torch.device("cuda"))
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2).to(torch.device("cuda"))
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2).to(torch.device("cuda"))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2).to(torch.device("cuda"))

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        #dec4 = self.up(1024, 512, kernel_size=2, stride=2)(bottleneck)
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        #dec3 = self.up(256, 256, kernel_size=2, stride=2)(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        #dec2 = self.up(128, 128, kernel_size=2, stride=2)(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        #dec1 = self.up(64, 64, kernel_size=2, stride=2)(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return self.output(dec1)
