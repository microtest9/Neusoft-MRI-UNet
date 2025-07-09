from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os, cv2, random, glob, torch

# ====================== 1. 数据预处理模块 ======================
class BrainMRIDataset(Dataset):
    """
    脑部MRI数据集加载器 
    功能：
    - 自动匹配图像和对应的掩码文件 
    - 过滤无效的空白掩码 
    - 支持训练集/验证集划分 
    - 返回归一化后的张量数据 
    """
    def __init__(self, data_dir, transform=None, train=True, train_ratio=0.8):
        """
        初始化数据集 
        参数:
            data_dir: str - 数据根目录路径 
            transform: torchvision.transform  - 数据增强变换 
            train: bool - 是否为训练集 
            train_ratio: float - 训练集划分比例 
        """
        self.data_dir  = data_dir 
        self.transform  = transform 
        self.train  = train 
        
        # 初始化路径列表 
        self.image_paths  = []  # 存储图像路径 
        self.mask_paths  = []   # 存储掩码路径 
        
        # 遍历数据目录中的所有子目录（每个子目录代表一个患者）
        for subdir in glob.glob(os.path.join(data_dir,  '*')):
            if os.path.isdir(subdir):   # 确保是目录 
                # 获取目录下所有文件 
                files = os.listdir(subdir) 
                
                # 分离图像文件和掩码文件（掩码文件名包含'mask'）
                images = [f for f in files if 'mask' not in f]  # 图像文件 
                masks = [f for f in files if 'mask' in f]      # 掩码文件 
                
                # 确保图像和掩码成对匹配 
                for img in images:
                    # 生成对应的掩码文件名（假设命名规则为：image.tif  和 image_mask.tif ）
                    mask = img.split('.')[0]  + '_mask.tif' 
                    if mask in masks:  # 如果对应的掩码存在 
                        self.image_paths.append(os.path.join(subdir,  img))
                        self.mask_paths.append(os.path.join(subdir,  mask))
        
        # 过滤掉掩码全为0的无效样本 
        self.valid_indices  = []
        for i, mask_path in enumerate(self.mask_paths): 
            # 使用OpenCV读取掩码（灰度模式）
            mask = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)
            if np.max(mask)  > 0:  # 检查掩码是否包含非零值（即有肿瘤区域）
                self.valid_indices.append(i)   # 只保留有效样本的索引 
        
        # 划分训练集和验证集 
        random.seed(42)   # 固定随机种子保证可复现性 
        num_samples = len(self.valid_indices) 
        indices = list(range(num_samples))
        random.shuffle(indices)   # 打乱顺序 
        
        # 按比例划分 
        split = int(train_ratio * num_samples)
        self.indices  = indices[:split] if train else indices[split:]
    
    def __len__(self):
        """返回数据集的有效样本数量"""
        return len(self.indices) 
    
    def __getitem__(self, idx):
        """
        获取单个样本 
        返回:
            image: Tensor [3, H, W] - 归一化的RGB图像 
            mask: Tensor [H, W] - 二值化掩码(0=背景, 1=肿瘤)
        """
        # 通过有效索引获取实际数据位置 
        real_idx = self.valid_indices[self.indices[idx]] 
        
        # 使用PIL加载图像和掩码 
        image = Image.open(self.image_paths[real_idx]).convert('RGB')   # 确保3通道 
        mask = Image.open(self.mask_paths[real_idx]).convert('L')        # 单通道灰度 
        
        # 应用数据增强（如果定义了transform）
        if self.transform: 
            image = self.transform(image)   # 转为Tensor并归一化 [3, H, W]
            mask = self.transform(mask)     # 转为Tensor [1, H, W]
        
        # 将掩码转为0/1二值图，并移除通道维度 [H, W]
        mask = (mask > 0).long().squeeze(0)
        
        return image, mask 