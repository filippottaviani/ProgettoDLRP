# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F # prova

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
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Mappa finale di profondità tra -1 e 1
        )

    def forward(self, x):
        # global
        out_enc = self.encoder(x)
        out_dec = self.decoder(out_enc)

        # prova dimensioni
        print(x.shape)
        print(out_dec.shape)

        # Verifica se le dimensioni di out_dec e x sono uguali, se no, usa F.interpolate
        if x.size(2) != out_dec.size(2) or x.size(3) != out_dec.size(3):
            out_dec = F.interpolate(out_dec, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)


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
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512)
        )

        # layer completamente connessi
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),  # assume dimensioni finali 4x4 dopo convoluzioni
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),  # output binario (vero/falso)
            nn.Sigmoid()
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
