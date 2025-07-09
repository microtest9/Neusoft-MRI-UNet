# ====================== 导入所需的库 ======================
# import matplotlib.pyplot as plt  # 可视化
import torch  # PyTorch主库
from torch.utils.data  import DataLoader  # 数据加载

from module_src.train import train_epoch
from module_src.evaluate import validate
from module_src.visualize import *

# ====================== 简易功能函数 ======================
# 伪回调
def dict_value_update(target: dict, values: list):
    for it_key in target:
        target.get(it_key).append(values.pop())

# 数据加载模板
# @workers 加载线程数
def get_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,batch_size=batch_size,
        shuffle=shuffle,num_workers=workers
    )

# 日志函数
def logc(stdop, c='blue'):
    if c=='blue':
        print(f'\033[0;37;44m{stdop}\033[0m')
    elif c=='red':
        print(f'\033[1;37;41m{stdop}\033[0m')
    elif c=='info':
        print(f'\033[4;33;40m{stdop}\033[0m')

# 执行过程
def run(rates: dict, *args, **kwargs):
    (best_dice,) = args
    val_loader = kwargs.pop('val_loader')
    train_params = kwargs
    for epoch in range(1,kwargs['epoch']+1):
        logc(f"Epoch {epoch}/{kwargs['epoch']}")
        # 训练阶段
        train_loss, train_dice = train_epoch(train_params)
        # 验证阶段
        val_loss, val_dice = validate(
            kwargs['model'], kwargs['device'],
            val_loader, kwargs['criterion']
        )
        # 实时输出
        print(f'Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Dice: {val_dice:.4f}')
        # 保存最优模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(kwargs['model'].state_dict(), 'pth_model/best_model.pth')
            logc("Saved best model!", 'info')
        logc(f"{'-' * 100}\n")
        # 记录指标
        dict_value_update(rates, [val_dice, train_dice, val_loss, train_loss])

    logc(f'Best Dice: {best_dice:.4f}', 'red')
    # ===== 可视化结果 ===== 
    visualize_results(kwargs['model'], kwargs['device'], val_loader)
    plot_metrics(rates)
    