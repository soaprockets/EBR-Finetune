import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
import dask.dataframe as dd
import os 

# 配置 Matplotlib 使用支持中文的字体并正常显示负号
# 尝试多种常见中文字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'STHeiti', 'STSong']
available_fonts = [f.name for f in font_manager.fontManager.ttflist]
chinese_font = None
for font in chinese_fonts:
    if font in available_fonts:
        chinese_font = font
        break

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font]
else:
    # 如果找不到中文字体，使用英文提示
    print("Warning: No Chinese font found, using English labels")
plt.rcParams['axes.unicode_minus'] = False

data_path = ''
filename = os.listdir(data_path)[0]
data_path = os.path.join(data_path, filename)
print(data_path)
df  = dd.read_csv(data_path, sep='\001', header=None, names=[])
df = df.compute()
# df = df.loc[~df.index.duplicated(keep='first')]
df['vector'] = df['vector'].apply(lambda x: [float(i) for i in x.split(' ')])
embeddings = np.array(df['vector'].tolist())

# 统一指定标签列，
label_column = ''

# 使用TSNE进行降维
tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)
tsne_2d = tsne.fit_transform(embeddings)

# 使用UMAP进行降维
umap_model = umap.UMAP(n_components=2, random_state=0)
umap_2d = umap_model.fit_transform(embeddings)


def compute_cluster_metrics(x, labels):
    """计算聚类质量指标"""
    metrics = {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or x.shape[0] <= len(unique_labels):
        return metrics
    try:
        metrics["silhouette"] = silhouette_score(x, labels)
    except ValueError:
        pass
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(x, labels)
    except ValueError:
        pass
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(x, labels)
    except ValueError:
        pass
    return metrics


def format_metrics_text(name, metrics):
    def fmt(value):
        return "nan" if np.isnan(value) else f"{value:.3f}"
    return (
        f"{name}\n"
        f"Silhouette: {fmt(metrics['silhouette'])}\n"
        f"C-H: {fmt(metrics['calinski_harabasz'])}\n"
        f"D-B: {fmt(metrics['davies_bouldin'])}"
    )



metrics_original = compute_cluster_metrics(embeddings, df[label_column])
metrics_tsne = compute_cluster_metrics(tsne_2d, df[label_column])
metrics_umap = compute_cluster_metrics(umap_2d, df[label_column])

# 手动指定颜色映射
unique_labels = df[label_column].unique()
print(unique_labels)

num_labels = len(unique_labels)
colors = sns.color_palette("husl", num_labels)  # 使用 husl 调色板生成颜色
color_map = {label: color for label, color in zip(unique_labels, colors)}

# 绘制 TSNE 与 UMAP 对比散点图并保存为图像数组
def plot_tsne_umap(tsne_values, umap_values, label):
    """绘制 t-SNE 与 UMAP 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    projections = [
        ("t-SNE Projection", tsne_values, metrics_tsne),
        ("UMAP Projection", umap_values, metrics_umap)
    ]
    
    for ax, (title, values, metric_values) in zip(axes, projections):
        tmp_df = pd.DataFrame(values, columns=['x', 'y'])
        tmp_df['label'] = label
        sns.scatterplot(
            x="x", y="y", hue="label", data=tmp_df, ax=ax,
            palette=color_map, alpha=0.6, s=50, legend=False
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02, 0.98,
            format_metrics_text(title, metric_values),
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )
    
    # 在图下方显示原始空间指标
    fig.text(
        0.5, 0.02,
        format_metrics_text("Original Embeddings", metrics_original),
        ha='center', va='bottom',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8)
    )
    fig.text(
        0.5, 0.95,
        "指标解读：Silhouette/CH 越大越好，Davies-Bouldin 越小越好",
        ha='center', va='top',
        fontsize=12,
        color='dimgray'
    )
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.imsave('./qwen_scatter.png', img)
    plt.close(fig)

plot_tsne_umap(tsne_2d, umap_2d, df[label_column])
