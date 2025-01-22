# models.py
import torch
import torch.nn as nn
from torchvision.models import vgg16

class GlobalNet1(nn.Module):
    def __init__(self):
        super(GlobalNet1, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        out_enc = self.encoder(x)
        out_dec = self.decoder(out_enc)

        return out_dec


class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()

        #refinement
        self.refinement1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.refinement2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        # additional
        self.additional = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, out_global, x):
        ref1_output = self.refinement1(out_global)
        #print(ref1_output.shape )
        add = self.additional(x)
        concat = torch.cat((add, ref1_output), dim=1)
        ref2_output = self.refinement2(concat)  # concatenazione tra uscita global e uscita ref

        return torch.add(out_global, ref2_output)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # elaborazione RGB
        self.rgb_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        # elaborazione Depth
        self.depth_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        # concatenazione e ulteriori convoluzioni
        self.joint_path = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # layer completamente connessi
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.LazyLinear(1024),
                                nn.LeakyReLU(0.2),
                                nn.Linear(1024, 1),
                                nn.Sigmoid())  # output tra 1 e 0 (vero/falso)

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
class GlobalNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(GlobalNet, self).__init__()

        # Blocchi di encoding
        self.enc1 = self.block(in_channels, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        self.enc4 = self.block(256, 512)

        # Bottleneck
        self.bottleneck = self.block(512, 1024)

        # Blocchi di decoding
        self.dec4 = self.block(1024, 512)
        self.dec3 = self.block(512, 256)
        self.dec2 = self.block(256, 128)
        self.dec1 = self.block(128, 64)

        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling e upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.up = nn.ConvTranspose2d non viene caricato correttamente in GPU
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2).to(torch.device("cuda"))
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2).to(torch.device("cuda"))
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2).to(torch.device("cuda"))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2).to(torch.device("cuda"))

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        # dec4 = self.up(1024, 512, kernel_size=2, stride=2)(bottleneck)
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        # dec3 = self.up(256, 256, kernel_size=2, stride=2)(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        # dec2 = self.up(128, 128, kernel_size=2, stride=2)(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        # dec1 = self.up(64, 64, kernel_size=2, stride=2)(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return self.output(dec1)


class GlobalNet2(nn.Module):
    def __init__(self):
        super(GlobalNet2, self).__init__()

        # carico il modello VGG16 pre-addestrato
        vgg = vgg16(weights=True)
        features = vgg.features # rimuovo gli ultimi livelli completamente connessi

        self.encoder = features
        for param in self.encoder.parameters():
            param.requires_grad = False  # Congela i layer

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=(3,2) , padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=(2,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=(2,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=(3,1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print(x.size())
        x = self.decoder(x)
        #print(x.size())
        return x