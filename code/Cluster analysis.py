from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
import numpy as np



# 读取数据
file_path = "/home/xqw/聚类分析/xu1.xlsx"
data = pd.read_excel(file_path)

# 选择特征列（假设要聚类的特征从第二列开始）
features = data.iloc[:, 0:].select_dtypes(include=[float, int])

# 对数据进行标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# 使用 t-SNE 进行非线性降维，调整参数
tsne = TSNE(n_components=2, perplexity=35, learning_rate=150, max_iter=5000, random_state=42)


X_tsne = tsne.fit_transform(scaled_features)

# 定义簇数范围
cluster_range = range(2, 10)
sse_scores = []
ch_scores = []
silhouette_scores = []

# 遍历簇数，计算SSE、CH指数和轮廓系数
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_tsne)

    # 计算SSE
    sse = kmeans.inertia_
    sse_scores.append(sse)

    # 计算CH指数
    ch_score = calinski_harabasz_score(X_tsne, labels)
    ch_scores.append(ch_score)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_tsne, labels)
    silhouette_scores.append(silhouette_avg)

# 使用 SSE 和 CH 确定最佳簇数
best_sse_clusters = cluster_range[sse_scores.index(min(sse_scores))]
best_ch_clusters = cluster_range[ch_scores.index(max(ch_scores))]

# 综合选择最佳簇数
best_n_clusters = best_ch_clusters if best_ch_clusters == best_sse_clusters else min(best_sse_clusters, best_ch_clusters)

# 使用最佳簇数进行最终的 K-Means 聚类
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(X_tsne)
final_kmeans_silhouette_score = silhouette_score(X_tsne, kmeans_labels)

print(f'最佳聚类数 (SSE 方法): {best_sse_clusters}')
print(f'最佳聚类数 (CH 方法): {best_ch_clusters}')
print(f'最终选择的最佳聚类数: {best_n_clusters}')
print(f'K-Means 聚类的轮廓系数: {final_kmeans_silhouette_score:.3f}')


# 将聚类结果加入原数据
data['KMeans_Cluster'] = kmeans_labels

# 绘制 K-Means 的轮廓系数图
sample_silhouette_values = silhouette_samples(X_tsne, kmeans_labels)
y_lower = 10


# 修改后的轮廓系数图绘制代码
plt.figure(figsize=(10, 8), facecolor='none')  # 设置整体画布透明

# 填充轮廓（保持原有代码）
y_lower = 10
for i in range(best_n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = sns.color_palette("viridis", as_cmap=True)(float(i) / best_n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

# 关键修改：设置坐标轴区域透明
ax = plt.gca()
ax.set_facecolor('none')  # 设置坐标轴背景透明

# 设置坐标轴标签和刻度颜色
plt.xlabel("Silhouette Coefficient Values",
           fontsize=24,
           fontname="Times New Roman",
           color='black')  # 标签设为黑色
plt.ylabel("Cluster",
           fontsize=24,
           fontname="Times New Roman",
           color='black')

# 设置刻度颜色
plt.tick_params(axis="both",
                colors='black',  # 刻度设为黑色
                direction="in",
                length=6,
                width=2)

# 设置坐标轴边框颜色
for spine in ax.spines.values():
    spine.set_color('black')  # 边框设为黑色
    spine.set_linewidth(2)

# 绘制红色虚线（保持可见性）
plt.axvline(x=final_kmeans_silhouette_score,
            color="red",
            linestyle="--",
            linewidth=2)

# 保存透明图像（推荐PNG格式）
plt.savefig('/home/xqw/聚类分析/silhouette_full_transparent.png',
            transparent=True,  # 强制透明
            dpi=300,
            bbox_inches='tight')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import seaborn as sns

# 获取每个簇的样本数据
cluster_data = [X_tsne[kmeans_labels == i] for i in range(best_n_clusters)]

# 创建绘图
plt.figure(figsize=(8, 6))

# 为每个簇计算置信区间并绘制
for i, data in enumerate(cluster_data):
    # 计算均值和协方差矩阵
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # 计算椭圆的主轴和宽度/高度（假设95%的置信区间）
    v, w = np.linalg.eigh(cov)
    v = 2.45 * np.sqrt(v)  # 95%置信区间, 2.45 是标准正态分布的系数
    v[0] *= 2  # 放大宽度
    v[1] *= 2  # 放大高度
    angle = np.arctan(w[0, 1] / w[0, 0]) * 180 / np.pi  # 计算旋转角度

    # 绘制椭圆
    ell = Ellipse(mean, width=v[0], height=v[1], angle=angle, edgecolor='black', facecolor='none', linestyle='--')
    plt.gca().add_patch(ell)

    # 绘制散点图
    plt.scatter(data[:, 0], data[:, 1], label=f'Cluster {i}', s=30)

# 设置标题和标签
plt.title("K-Means Clustering with Confidence Intervals", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=14)
plt.ylabel("t-SNE Component 2", fontsize=14)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 样式
sns.set(style="whitegrid")

# 创建一个图形和两个子图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 SSE 曲线
ax1.plot(cluster_range, sse_scores, color='tab:red', marker='o', label='SSE', linewidth=2)
ax1.set_xlabel('Number of Clusters', fontsize=12)
ax1.set_ylabel('SSE (Sum of Squared Errors)', fontsize=12, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# 创建第二个 Y 轴
ax2 = ax1.twinx()

# 绘制 CH 曲线
ax2.plot(cluster_range, ch_scores, color='tab:blue', marker='o', label='CH Index', linewidth=2)
ax2.set_ylabel('Calinski-Harabasz Index', fontsize=12, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# 设置标题和图例
fig.tight_layout()

# 显示图表
plt.show()

# 保存 t-SNE 降维结果和聚类标签到 CSV 文件
tsne_cluster_data = pd.DataFrame(X_tsne, columns=['t-SNE Component 1', 't-SNE Component 2'])
tsne_cluster_data['KMeans Cluster'] = kmeans_labels
tsne_cluster_data.to_csv('/home/xqw/聚类分析/tsne_cluster_data0305.csv', index=False)
print("t-SNE 降维结果和聚类标签已保存为 tsne_cluster_data.csv")

# 提取轮廓系数数据
silhouette_data = pd.DataFrame({
    'Sample Index': np.arange(len(sample_silhouette_values)),
    'Silhouette Coefficient': sample_silhouette_values,
    'Cluster': kmeans_labels
})
silhouette_data.to_csv('/home/xqw/聚类分析/silhouette_data0305.csv', index=False)
print("Silhouette 图数据已保存为 silhouette_data.csv")

# 保存 SSE 和 CH 数据到 CSV 文件
sse_ch_data = pd.DataFrame({
    'Number of Clusters': list(cluster_range),
    'SSE': sse_scores,
    'CH Index': ch_scores
})
sse_ch_data.to_csv('/home/xqw/聚类分析/sse_ch_data0305.csv', index=False)
print("SSE 和 CH 指数数据已保存为 sse_ch_data.csv")

# 计算每个簇的均值、协方差、椭圆参数
confidence_data = []

for i, data in enumerate(cluster_data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # 计算椭圆的主轴和宽度/高度（假设95%的置信区间）
    v, w = np.linalg.eigh(cov)
    v = 2.45 * np.sqrt(v)  # 95%置信区间, 2.45 是标准正态分布的系数
    v[0] *= 2  # 放大宽度
    v[1] *= 2  # 放大高度
    angle = np.arctan(w[0, 1] / w[0, 0]) * 180 / np.pi  # 计算旋转角度

    # 保存数据
    confidence_data.append({
        'Cluster': i,
        'Mean X': mean[0],
        'Mean Y': mean[1],
        'Covariance XX': cov[0, 0],
        'Covariance XY': cov[0, 1],
        'Covariance YY': cov[1, 1],
        'Ellipse Width': v[0],
        'Ellipse Height': v[1],
        'Ellipse Angle': angle
    })

# 保存椭圆数据到 CSV 文件
confidence_df = pd.DataFrame(confidence_data)
confidence_df.to_csv('/home/xqw/聚类分析/confidence_ellipses0305.csv', index=False)
print("置信区间（椭圆）数据已保存为 confidence_ellipses.csv")
