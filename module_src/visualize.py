from torch import no_grad, sigmoid
import matplotlib.pyplot as plt
import numpy as np

# ====================== 4.数据和结果可视化 =====================
def plot_metrics(rates: dict):
    """绘制 Loss 和 Dice 曲线图"""
    train_losses, val_losses, train_dices, val_dices = [rates[key] for key in rates.keys()]
    plt.figure(figsize=(9, 4))
    plt.gcf().canvas.toolbar_visible=True

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.title('Dice Coefficient Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_results(model, device, val_loader, num_samples=3):
    """修正后的可视化函数"""
    model.eval() 
    with no_grad(): 
        # 获取一个batch的数据 
        images, masks = next(iter(val_loader))
        images = images.to(device)
        
        # 模型预测 [B,1,H,W] -> [B,H,W]
        outputs = model(images)
        preds = sigmoid(outputs).squeeze(1).cpu().numpy()   # 移除通道维度 
        preds = (preds > 0.5).astype(np.float32)   # 二值化并确保数据类型正确 
        
        # 转换其他数据格式 
        images_np = images.permute(0,  2, 3, 1).cpu().numpy()  # [B,H,W,C]
        masks_np = masks.cpu().numpy()   # [B,H,W]
        
        # 调试信息（可选）
        print(f"图像形状: {images_np.shape}")
        print(f"掩码形状: {masks_np.shape}")
        print(f"预测形状: {preds.shape}")
        
        # 可视化 
        plt.figure(figsize=(9,  2*num_samples))
        plt.gcf().canvas.toolbar_visible=True
        for i in range(num_samples):
            # 原始图像 [H,W,C]
            plt.subplot(num_samples,  3, i*3+1)
            plt.imshow(images_np[i]) 
            plt.title('Input  Image')
            plt.axis('off') 
            
            # 真实掩码 [H,W]
            plt.subplot(num_samples,  3, i*3+2)
            plt.imshow(masks_np[i],  cmap='gray', vmin=0, vmax=1)
            plt.title('Ground  Truth')
            plt.axis('off') 
            
            # 预测结果 [H,W]
            plt.subplot(num_samples,  3, i*3+3)
            plt.imshow(preds[i],  cmap='gray', vmin=0, vmax=1)
            plt.title('Prediction') 
            plt.axis('off') 
        
        plt.tight_layout() 
        plt.show()