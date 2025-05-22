import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleEncoderBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels)
        )

        self.feature_fusion = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x3 = self.conv_block3(x)

        x_cat = torch.cat((x1, x2, x3), dim=1)

        out = self.feature_fusion(x_cat)

        return out

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet3D, self).__init__()
        #self.multi_scale_enc1 = MultiScaleEncoderBlock(1, 32)
        self.enc1 = self.contract_block(in_channels, 32)
        #self.multi_scale_enc2 = MultiScaleEncoderBlock(32, 64)
        self.enc2 = self.contract_block(32, 64)
        #self.multi_scale_enc3 = MultiScaleEncoderBlock(64, 128)
        self.enc3 = self.contract_block(64, 128)
        #self.multi_scale_enc4 = MultiScaleEncoderBlock(128, 256)
        self.enc4 = self.contract_block(128, 256)
        self.bottleneck = self.contract_block(256, 512, apply_pooling=False)
        self.dec4 = self.expand_block(512 + 256, 256)
        self.dec3 = self.expand_block(256 + 128, 128)
        self.dec2 = self.expand_block(128 + 64, 64)
        self.dec1 = self.expand_block(64 + 32, 32)
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        #enc1 = self.multi_scale_enc1(x)
        enc1 = self.enc1(x)
        #enc2 = self.multi_scale_enc2(enc1)
        enc2 = self.enc2(enc1)
        #enc3 = self.multi_scale_enc3(enc2)
        enc3 = self.enc3(enc2)
        #enc4 = self.multi_scale_enc4(enc3)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        dec4 = self.dec4(torch.cat((bottleneck, enc4), dim=1))
        dec3 = self.dec3(torch.cat((dec4, enc3), dim=1))
        dec2 = self.dec2(torch.cat((dec3, enc2), dim=1))
        dec1 = self.dec1(torch.cat((dec2, enc1), dim=1))
        final_output = self.final(dec1)
        return final_output

    def contract_block(self, in_channels, out_channels, apply_pooling=True):
        layers = [
            #MultiScaleEncoderBlock(in_channels, out_channels),  # Use MultiScaleEncoderBlock
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels)
        ]
        if apply_pooling:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.BatchNorm3d(out_channels)
        )
