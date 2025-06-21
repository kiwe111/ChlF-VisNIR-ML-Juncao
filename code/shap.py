import shap
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

file_path = "/home/xqw/0326.xlsx"
data = pd.read_excel(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练 XGBoost 分类模型
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=42)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap_array = shap_values.values

# 计算每个类别的 SHAP 贡献中心
shap_cluster_centers = np.zeros((3, X.shape[1]))
for cluster in range(3):
    cluster_indices = np.where(np.array(y_test) == cluster)[0]  # 选出该簇的样本索引
    shap_cluster_centers[cluster] = np.mean(shap_array[cluster_indices, :, cluster], axis=0)

# 转换为 DataFrame
shap_df = pd.DataFrame(shap_cluster_centers.T, index=data.columns[:-1], columns=['NC', 'MC', 'SC'])

# 绘制 SHAP 贡献柱状图
shap_df.plot(kind='bar', figsize=(14, 6), width=0.8, colormap='Reds')
plt.ylabel("Mean SHAP Contribution")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.legend(title="Cluster")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 计算每个特征的 SHAP 贡献总和（取绝对值后求和）
shap_df['Total_Contribution'] = shap_df.abs().sum(axis=1)

# 按照 SHAP 贡献总和排序
shap_df_sorted = shap_df.sort_values(by='Total_Contribution', ascending=False).drop(columns=['Total_Contribution'])

# 绘制堆叠柱状图
plt.figure(figsize=(14, 6))
shap_df_sorted.abs().plot(kind='bar', stacked=True, width=0.8, figsize=(14, 6),
                          color=['darkblue', 'darkred', 'darkgreen'], alpha=0.85)
plt.ylabel("Total SHAP Contribution (Absolute Sum)")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Cluster", labels=['NC', 'MC', 'SC'])  # 重新设置标签
plt.title("Stacked SHAP Contribution per Feature (Sorted by Total Contribution)")
plt.show()

# 计算 SHAP 值
shap_values_full = explainer(X_test)

# 绘制 SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_full, X_test, feature_names=data.columns[:-1], show=True)


# 计算每个特征与标签之间的皮尔逊相关系数
correlation_with_target = X.corrwith(y)

# 将相关系数按绝对值排序
correlation_with_target_sorted = correlation_with_target.abs().sort_values(ascending=False)

# 可视化皮尔逊相关性
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target_sorted.index, y=correlation_with_target_sorted.values)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Pearson Correlation with Target")
plt.title("Pearson Correlation between Features and Target")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 生成分类报告（包括 Precision, Recall, F1-score）
class_report = classification_report(y_test, y_pred, target_names=['NC', 'MC', 'SC'])

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'MC', 'SC'], yticklabels=['NC', 'MC', 'SC'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 输出评估结果
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(class_report)

# 将结果导出为CSV文件
correlation_with_target_sorted.to_csv('/home/xqw/pearson_correlation.csv', header=True)

print("皮尔逊相关性数据已导出到 'pearson_correlation.csv'")

# 导出 SHAP 数据到 Excel 文件
shap_df_sorted.to_excel("/home/xqw/shap_contribution_sorted2.xlsx", sheet_name="SHAP_Contribution_Sorted")

# 导出原始 SHAP 贡献中心到 Excel 文件
shap_df.to_excel("/home/xqw/shap_contribution_centers2.xlsx", sheet_name="SHAP_Contribution_Centers")


