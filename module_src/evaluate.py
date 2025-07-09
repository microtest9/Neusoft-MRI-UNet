from torch import no_grad, sigmoid
from tqdm import tqdm as progress

def dice_coeff(pred, target, smooth=1.0):
    """
    计算Dice系数(分割任务评估指标)
    公式: (2*|X∩Y|) / (|X| + |Y|)
    
    参数:
        pred: Tensor - 模型预测结果（二值化后）
        target: Tensor - 真实标签 
        smooth: float - 平滑系数(避免除以0)
    """
    # 展平预测和标签 
    pred_flat = pred.contiguous().view(-1) 
    target_flat = target.contiguous().view(-1) 
    
    # 计算交集 
    intersection = (pred_flat * target_flat).sum()
    
    # 计算Dice系数 
    return (2. * intersection + smooth) / (pred_flat.sum()  + target_flat.sum()  + smooth)

def validate(model, device, val_loader, criterion):
    """
    验证模型 
    
    参数:
        model: nn.Module - 待验证模型 
        device: torch.device  - 计算设备 
        val_loader: DataLoader - 验证数据加载器 
        criterion: nn.Module - 损失函数 
    """
    model.eval()   # 设置为评估模式 
    val_loss = 0.0 
    val_dice = 0.0 
    
    with no_grad():   # 禁用梯度计算 
        for images, masks in progress(val_loader, desc='Validating'):
            images = images.to(device) 
            masks = masks.to(device).unsqueeze(1)   # 添加通道维度 
            
            # 前向传播 
            outputs = model(images)
            
            # 计算损失 
            val_loss += criterion(outputs, masks.float()).item() 
            
            # 计算Dice 
            preds = sigmoid(outputs)  > 0.5
            val_dice += dice_coeff(preds, masks).item()
    
    # 计算平均指标 
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    
    return val_loss, val_dice