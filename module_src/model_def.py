from torch import cat
import torch.nn as nn

# ====================== 2. U-Net模型定义 ======================
class DoubleConv(nn.Module):
    """
    双卷积块(卷积+BN+ReLU)*2 
    用于U-Net的编码器和解码器路径 
    """
    def __init__(self, in_channels, out_channels):
        """
        参数:
            in_channels: int - 输入通道数 
            out_channels: int - 输出通道数 
        """
        super().__init__()
        self.double_conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.double_conv(x) 

class Down(nn.Module):
    """
    下采样块（最大池化+双卷积）
    用于U-Net的编码器路径 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv  = nn.Sequential(
            nn.MaxPool2d(2),  # 2×2最大池化，尺寸减半 
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x) 
 
class Up(nn.Module):
    """
    上采样块（转置卷积+双卷积）
    用于U-Net的解码器路径 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用转置卷积进行上采样 
        self.up  = nn.ConvTranspose2d(
            in_channels, in_channels // 2,  # 通道数减半 
            kernel_size=2, stride=2 
        )
        self.conv  = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        参数:
            x1: 上采样特征 
            x2: 跳跃连接特征 
        """
        x1 = self.up(x1)   # 上采样 
        
        # 处理尺寸不匹配问题（由于池化导致的奇数尺寸）
        diffY = x2.size()[2]  - x1.size()[2] 
        diffX = x2.size()[3]  - x1.size()[3] 
        
        # 对称填充 
        x1 = nn.functional.pad( 
            x1, 
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        
        # 沿通道维度拼接特征 
        x = cat([x2,  x1], dim=1)
        return self.conv(x) 
 
class UNet(nn.Module):
    """
    U-Net模型主结构 
    经典的编码器-解码器架构，带有跳跃连接 
    """
    def __init__(self, n_channels=3, n_classes=1):
        """
        参数:
            n_channels: int - 输入图像通道数（默认3，RGB）
            n_classes: int - 输出类别数（默认1，二分类）
        """
        super().__init__()
        self.n_channels = n_channels 
        self.n_classes = n_classes 
        
        # ===== 编码器路径（下采样） ===== 
        self.inc  = DoubleConv(n_channels, 64)  # 初始卷积 
        self.down1  = Down(64, 128)    # 第一次下采样 
        self.down2  = Down(128, 256)   # 第二次下采样 
        self.down3  = Down(256, 512)   # 第三次下采样 
        self.down4  = Down(512, 1024)  # 第四次下采样 
        
        # ===== 解码器路径（上采样） ===== 
        self.up1  = Up(1024, 512)  # 第一次上采样 
        self.up2  = Up(512, 256)   # 第二次上采样 
        self.up3  = Up(256, 128)   # 第三次上采样 
        self.up4  = Up(128, 64)    # 第四次上采样 
        
        # 输出层（1×1卷积）
        self.outc  = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # ===== 编码器 ===== 
        x1 = self.inc(x)     # 初始特征 [B,64,H,W]
        x2 = self.down1(x1)  # [B,128,H/2,W/2]
        x3 = self.down2(x2)  # [B,256,H/4,W/4]
        x4 = self.down3(x3)  # [B,512,H/8,W/8]
        x5 = self.down4(x4)  # [B,1024,H/16,W/16] 瓶颈层 
        
        # ===== 解码器 ===== 
        x = self.up1(x5,  x4)  # [B,512,H/8,W/8]
        x = self.up2(x,  x3)   # [B,256,H/4,W/4]
        x = self.up3(x,  x2)   # [B,128,H/2,W/2]
        x = self.up4(x,  x1)   # [B,64,H,W]
        
        # 输出层 
        logits = self.outc(x)   # [B,1,H,W]
        return logits 
 