import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
import os
import seaborn as sns
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
matplotlib.rcParams['pdf.fonttype'] = 42

# plt.rcParams['font.family'] = 'SimHei'

def collect_lora_matrices(model):
    lora_dict = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            layer_key = name.replace('.lora_A', '')
            if layer_key not in lora_dict:
                lora_dict[layer_key] = {}
            lora_dict[layer_key]['A'] = param.detach().cpu().numpy()
        elif 'lora_B' in name:
            layer_key = name.replace('.lora_B', '')
            if layer_key not in lora_dict:
                lora_dict[layer_key] = {}
            lora_dict[layer_key]['B'] = param.detach().cpu().numpy()
    return lora_dict

def plot_lora_heatmaps(lora_dict, figsize_per_subplot=(3, 2), cmap='coolwarm'):
    layers = list(lora_dict.keys())
    n_layers = len(layers)
    n_subplots = n_layers
    cols = math.ceil(math.sqrt(n_subplots))
    rows = math.ceil(n_subplots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_per_subplot[0], rows*figsize_per_subplot[1]))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # 统一颜色范围
    all_values = []
    for data in lora_dict.values():
        all_values.extend(data.flatten())
    vmin, vmax = min(all_values), max(all_values)
    
    for i, (layer_name, value) in enumerate(lora_dict.items()):
        ax_idx = i
        ax = axes[ax_idx // cols, ax_idx % cols]
        
        sns.heatmap(value, ax=ax, cmap=cmap, center=0,
                    vmin=vmin, vmax=vmax, cbar=False,
                    xticklabels=False, yticklabels=False)
        if 'lora_B' in layer_name:
            ax.set_title(f"lora_B ({value.shape[0]}×{value.shape[1]})", fontsize=8)
        else:
            ax.set_title(f"lora_A ({value.shape[0]}×{value.shape[1]})", fontsize=8)
    
    # 隐藏多余子图
    for j in range(n_subplots, rows*cols):
        axes[j // cols, j % cols].axis('off')
    
    # 添加全局颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Value')
    
    plt.suptitle('LoRA Matrices Heatmaps (All Layers)', y=0.98)
    plt.show()

path = 'distilbert-base-multilingual-cased_'
file = path + str(19) + '.pt'
param = torch.load(file)
plot_lora_heatmaps(param, figsize_per_subplot=(2.5, 2))
# fig, axes = plt.subplots(3, 2, figsize=(8, 9))
# ax_idx = 0
# for i in range(19, 20, 1):
    
#     idx = 0
#     for k, v in name_grad.items():
#         if idx % 2 == 0:
#             lora_A = v
#             # result = torch.norm(lora_A, dim=0) ** 2
#             # # # result = lora_B @ lora_A
#             # result, _ = torch.sort(result.view(-1).abs(), descending=True)
#             # x = range(1, result.numel()+1)
#             # ax1.plot(x, result, label='real')

#             # #  拟合幂律分布
#             # def power(x, alpha, c):
#             #     # return c * np.log(alpha*x)
#             #     # return c * np.exp(-alpha*x)
#             #     return c * np.power(x, alpha)
#             # def log(x, alpha, d):
#             #     return -0.01*np.log(alpha*x) + d
#             # def exp(x, alpha):
#             #     return 0.001*np.exp(-alpha*x)
            
#             # x = torch.tensor(range(1, result.numel() + 1))
#             # powert, _ = curve_fit(power, x.numpy(), result.numpy())
#             # ax1.plot(x, powert[1]*np.power(x, powert[0]), label='fitted by power law', linestyle='--', marker='^', markevery=50)
#             # logt, _ = curve_fit(log, x.numpy(), result.numpy())
#             # ax1.plot(x, -0.01*np.log(x * logt[0]) + logt[1], label='fitted by log', linestyle='--', marker='.', markevery=50)
#             # expt, _ = curve_fit(exp, x.numpy(), result.numpy())
#             # ax1.plot(x, 0.001*np.exp(-x * expt[0]), label='fitted by exp', linestyle='--', marker='*', markevery=50)
#             # ax1.legend(loc='best')
#             # ax1.set_xlabel('rank ID', size=12)
#             # ax1.set_ylabel('value', size=12)
#             # ax1.set_title("low-rank matrix $\mathbf{A}$", y=-0.3)
#         else:
#             lora_B = v
#             # result = torch.norm(lora_B, dim = 1) ** 2
#             # # result = lora_B @ lora_A
#             # result, _ = torch.sort(result.view(-1).abs(), descending=True)
#             # x = range(1, result.numel()+1)
#             # ax2.plot(x, result, label='real')

#             # #  拟合幂律分布
#             # def power(x, alpha, c):
#             #     # return c * np.log(alpha*x)
#             #     # return c * np.exp(-alpha*x)
#             #     return c * np.power(x, alpha)
#             # def log(x, alpha, d):
#             #     return -0.0001*np.log(alpha*x) + d
#             # def exp(x, alpha):
#             #     return 0.00001*np.exp(-alpha*x)
            
#             # x = torch.tensor(range(1, result.numel() + 1))
#             # powert, _ = curve_fit(power, x.numpy(), result.numpy())
#             # ax2.plot(x, powert[1]*np.power(x, powert[0]), label='fitted by power law', linestyle='--', marker='^', markevery=50)
#             # logt, _ = curve_fit(log, x.numpy(), result.numpy())
#             # ax2.plot(x, -0.0001*np.log(x * logt[0]) + logt[1], label='fitted by log', linestyle='--', marker='.', markevery=50)
#             # expt, _ = curve_fit(exp, x.numpy(), result.numpy())
#             # ax2.plot(x, 0.00001*np.exp(-x * expt[0]), label='fitted by exp', linestyle='--', marker='*', markevery=50)
#             # ax2.legend(loc='best')
#             # ax2.set_xlabel('rank ID', size=12)
#             # ax2.set_ylabel('value', size=12)
#             # ax2.set_title("low-rank matrix $\mathbf{B}$", y=-0.3)
#             # 绘制第一个热力图，不显示颜色条（cbar=False）
#             vmin = min(lora_B.min(), lora_A.min())
#             vmax = max(lora_B.max(), lora_A.max())
#             sns.heatmap(lora_B, ax=axes[ax_idx], cmap='coolwarm', center=0,
#                         vmin=vmin, vmax=vmax, cbar=False,
#                         xticklabels=False, yticklabels=False)
#             axes[ax_idx].set_title(f'Matrix B')
#             axes[ax_idx].set_xlabel('Rank dimension')
#             axes[ax_idx].set_ylabel('Hidden dimension')

#             ax_idx += 1
#             sns.heatmap(lora_A, ax=axes[ax_idx], cmap='coolwarm', center=0,
#                         vmin=vmin, vmax=vmax, cbar=False,
#                         xticklabels=False, yticklabels=False)
#             axes[ax_idx].set_title(f'Matrix A')
#             axes[ax_idx].set_xlabel('Hidden dimension')
#             axes[ax_idx].set_ylabel('Rank dimension')
#             ax_idx += 1
#             if idx == 5:
#                 break
#         idx += 1

# plt.tight_layout()
# plt.show()