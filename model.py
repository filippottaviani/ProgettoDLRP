# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F # prova

class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=1, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=2), nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        # global
        out_enc = self.encoder(x)
        out_dec = self.decoder(out_enc)

        return out_dec


class RefinementNet(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Mappa finale di profondità tra -1 e 1
        )

        self.refinement2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()  # Mappa finale di profondità tra -1 e 1
        )

    def forward(self, x, out_global):
        # upsampling
        up = self.upsample(out_global)

        # refinement
        ref1_output = self.refinement1(up)
        add = self.additional(x)
        concat = torch.cat((add, ref1_output), dim=1)
        ref2_output = self.refinement2(concat)

        # print("refinement: ", ref_output.size())

        # concatenazione tra uscita glob(x2) e uscita ref
        combined_output = torch.add(up, ref2_output)

        # print("out: ", combined_output.size())

        return combined_output




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
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # conv2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # conv3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # conv4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

        )

        # layer completamente connessi
        self.fc = nn.Sequential(nn.Flatten(), nn.LazyLinear(1024), nn.LeakyReLU(0.2),
                                nn.Linear(1024, 1), nn.Sigmoid())  # output tra 1 e 0 (vero/falso)

    def forward(self, rgb, depth):
        rgb_features = self.rgb_path(rgb)
        depth_features = self.depth_path(depth)

        # concatenazione lungo il canale
        joint_features = torch.cat((rgb_features, depth_features), dim=1)
        joint_output = self.joint_path(joint_features)

        # passaggio al livello completamente connesso
        output = self.fc(joint_output)
        return output
