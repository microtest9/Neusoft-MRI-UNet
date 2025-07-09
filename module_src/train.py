from tqdm import tqdm as progress
from torch import sigmoid
from module_src.evaluate import dice_coeff

def train_epoch(train_params: dict):
    """
    训练单个epoch 
    
    参数:
        model: nn.Module - 待训练模型 
        device: torch.device  - 计算设备(CPU/GPU)
        train_loader: DataLoader - 训练数据加载器 
        optimizer: torch.optim  - 优化器 
        criterion: nn.Module - 损失函数 
        epoch: int - 当前epoch数(用于显示)
    """
    loader = train_params['train_loader']
    train_params['model'].train()   # 设置为训练模式
    # 累计损失, 累计dice系数
    running_loss, running_dice = 0.0, 0.0
    
    # 使用tqdm显示进度条
    for images, masks in progress(loader, desc=f"Train Epoch {train_params['epoch']}"):
        # 将数据移动到设备
        images = images.to(train_params['device'])
        masks = masks.to(train_params['device']).unsqueeze(1)   # 添加通道维度 [B,1,H,W]

        # 梯度清零
        train_params['optimizer'].zero_grad()
        # 前向传播
        outputs = train_params['model'](images)
        # 计算损失
        loss = train_params['criterion'](outputs, masks.float())
        # 反向传播
        loss.backward()
        train_params['optimizer'].step()

        # 计算统计量
        running_loss += loss.item() 
        # 将输出转为二值预测并计算Dice
        preds = sigmoid(outputs)  > 0.5 
        running_dice += dice_coeff(preds, masks).item()
    
    # 计算平均损失和Dice
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    
    return epoch_loss, epoch_dice