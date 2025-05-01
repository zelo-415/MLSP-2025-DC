import torch
import torch.nn as nn
import torch.nn.functional as F
 
class DoubleConv(nn.Module):
     def __init__(self, in_channels, out_channels):
         super().__init__()
         self.double_conv = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )
 
     def forward(self, x):
         return self.double_conv(x)
 
class UNet(nn.Module):
     def __init__(self, in_channels=6, out_channels=1):  # 
         super().__init__()
         self.enc1 = DoubleConv(in_channels, 64)
         self.pool1 = nn.MaxPool2d(2)
         self.enc2 = DoubleConv(64, 128)
         self.pool2 = nn.MaxPool2d(2)
         self.enc3 = DoubleConv(128, 256)
         self.pool3 = nn.MaxPool2d(2)
 
         self.bottleneck = DoubleConv(256, 512)
 
         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
         self.dec3 = DoubleConv(512, 256)
         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
         self.dec2 = DoubleConv(256, 128)
         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
         self.dec1 = DoubleConv(128, 64)
 
         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
 
     def forward(self, x):
         e1 = self.enc1(x)
         e2 = self.enc2(self.pool1(e1))
         e3 = self.enc3(self.pool2(e2))
 
         b = self.bottleneck(self.pool3(e3))
 
         d3 = self.up3(b)
         d3 = self.dec3(torch.cat([d3, e3], dim=1))
         d2 = self.up2(d3)
         d2 = self.dec2(torch.cat([d2, e2], dim=1))
         d1 = self.up1(d2)
         d1 = self.dec1(torch.cat([d1, e1], dim=1))
 
         return self.final_conv(d1)

class SpatialTransformer(nn.Module):
    def __init__(self, input_channels=1):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 3, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Create placeholder; we'll init FC later
        self.fc_loc = None
        self.fc_loc_initialized = False

    def forward(self, x):
        xs = self.localization(x)

        xs = xs.view(xs.size(0), )
        grid = F.affine_grid(xs, x.size(), align_corners=True)
        x_transformed = F.grid_sample(x, grid, align_corners=True)
        return x_transformed    

class UNetWithSTN(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNetWithSTN, self).__init__()
        self.stn = SpatialTransformer(input_channels=in_channels)
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x):
        print("\n\n",x.shape)
        x_transformed = self.stn(x)
        print(x_transformed.shape)
        x_out = self.unet(x_transformed)
        return x_out