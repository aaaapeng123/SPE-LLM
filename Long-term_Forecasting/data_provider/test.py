import matplotlib.pyplot as plt
from matplotlib import font_manager

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 原始输入长度和对应横坐标位置
input_lengths = ['128', '256', '512', '1024']
x_positions = range(len(input_lengths))

# 数据
dataset_A = [0.456, 0.422, 0.39, 0.385]
dataset_B = [0.233, 0.214, 0.191, 0.187]
dataset_C = [0.265, 0.247, 0.234, 0.230]
dataset_D = [0.311, 0.289, 0.264, 0.262]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制每个数据集（增大线宽和点大小）
plt.plot(x_positions, dataset_A, label='ETTh1', marker='o', linestyle='-', color='blue', linewidth=4, markersize=12)
plt.plot(x_positions, dataset_B, label='Weather', marker='s', linestyle='--', color='red', linewidth=4, markersize=12)
plt.plot(x_positions, dataset_C, label='Electricity', marker='^', linestyle=':', color='green', linewidth=4, markersize=12)
plt.plot(x_positions, dataset_D, label='Traffic', marker='D', linestyle='-.', color='orange', linewidth=4, markersize=12)

# 设置图表标题和轴标签字体大小
plt.xlabel('Input Length', fontsize=24)
plt.ylabel('MAE', fontsize=24)

# 设置横纵坐标刻度字体大小
plt.xticks(x_positions, input_lengths, fontsize=20)
plt.yticks([0.18,0.23,0.28,0.33,0.38,0.43,0.48], fontsize=22)

# 设置 Y 轴的刻度间隔，从0.2到0.3，并去除中间的空白
# 设置 Y 轴范围 # 设置Y轴范围，紧贴数据

# 只显示横向网格线
plt.grid(True, axis='y')

# 添加图例到下方（居中且横向排列）
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=22, frameon=False)

# 布局调整与显示
plt.tight_layout()
plt.show()
