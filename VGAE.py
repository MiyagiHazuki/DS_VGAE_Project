import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE
import scanpy as sc
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import holoviews as hv
from holoviews import opts

# 读取和预处理数据
adata = sc.read_h5ad("./human_brain_region_88_sparse_with3d.h5ad")
sc.pp.normalize_total(adata, inplace=True)

# 提取空间特征和区域名称
spatial_features = adata.obs.iloc[:, -3:].values
region_names = adata.obs['region_name'].values

# 将稀疏矩阵转换为密集矩阵
dense_matrix = adata.X.toarray()

# 创建DataFrame以便聚合
df = pd.DataFrame(dense_matrix, index=region_names)
spatial_df = pd.DataFrame(spatial_features, index=region_names, columns=['x', 'y', 'z'])

# 对特征和空间数据进行平均聚合
aggregated_features = df.groupby(df.index).mean().values
aggregated_spatial = spatial_df.groupby(spatial_df.index).mean().values

# PCA降维
pca = PCA(n_components=88)  # 根据需要调整PCA的组件数
pca_features = pca.fit_transform(aggregated_features)

# 将PCA特征与空间特征结合
combined_features = np.hstack((pca_features, aggregated_spatial))

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 计算聚合后空间特征的欧式距离
# distances = squareform(pdist(aggregated_spatial, metric='euclidean'))

# # 定义阈值
# threshold = 3.0  # 根据需要调整阈值

# # 创建邻接矩阵
# adj_matrix = (distances < threshold).astype(int)

# # 打印原始邻接矩阵及其尺寸
# print("Original Adjacency Matrix:")
# print(adj_matrix)
# print(f"Original Adjacency Matrix Shape: {adj_matrix.shape}")

# # 创建边索引
# edge_index = np.array(np.nonzero(adj_matrix))
# edge_index = torch.tensor(edge_index, dtype=torch.long)

# # 将聚合后的特征转换为张量
# x = torch.tensor(combined_features, dtype=torch.float)

# # 创建图数据
# graph_data = Data(x=x, edge_index=edge_index).to(device)

# # 定义VGAE模型
# class VariationalGraphAutoencoder(VGAE):
#     def __init__(self, in_channels, out_channels):
#         super(VariationalGraphAutoencoder, self).__init__(encoder=GCNEncoder(in_channels, out_channels))

# class GCNEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(in_channels, 4 * out_channels)
#         self.bn1 = nn.BatchNorm1d(4 * out_channels)
#         self.dropout1 = nn.Dropout(p=0.6)  # 添加Dropout层
#         self.conv2 = GCNConv(4 * out_channels, 4 * out_channels)
#         self.bn2 = nn.BatchNorm1d(4 * out_channels)
#         self.dropout2 = nn.Dropout(p=0.6)  # 添加Dropout层
#         self.conv_mu = GCNConv(4 * out_channels, out_channels)
#         self.conv_logvar = GCNConv(4 * out_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = F.gelu(self.bn1(self.conv1(x, edge_index)))
#         x = self.dropout1(x)  # 应用Dropout
#         x = F.gelu(self.bn2(self.conv2(x, edge_index)))
#         x = self.dropout2(x)  # 应用Dropout
#         return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# # 实例化VGAE模型
# out_channels = 88
# vgae_model = VariationalGraphAutoencoder(in_channels=x.size(1), out_channels=out_channels).to(device)

# # 定义优化器，添加L2正则化
# optimizer = torch.optim.AdamW(vgae_model.parameters(), lr=0.01)  # 不使用L2正则化

# # 记录损失
# losses = []

# # 训练VGAE模型
# vgae_model.train()
# for epoch in range(5000):  # 增加训练轮次
#     optimizer.zero_grad()
#     z = vgae_model.encode(graph_data.x, graph_data.edge_index)
#     loss = vgae_model.recon_loss(z, graph_data.edge_index)
#     loss = loss + (1 / graph_data.num_nodes) * vgae_model.kl_loss()
#     loss.backward()
#     optimizer.step()
    
#     # 记录损失
#     losses.append(loss.item())
    
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")

# # 绘制损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Time')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_loss.png')
# plt.show()

# print("Training loss plot saved to 'training_loss.png'")

# # 获取更新后的边权重
# vgae_model.eval()
# with torch.no_grad():
#     z = vgae_model.encode(graph_data.x, graph_data.edge_index)
#     prob_adj = vgae_model.decoder.forward_all(z)

# # 使用较低的阈值来生成邻接矩阵
# threshold = 0.5
# updated_adj_matrix = (prob_adj > threshold).int().cpu().numpy()

# # 删除自环
# np.fill_diagonal(updated_adj_matrix, 0)

# # 保存更新后的邻接矩
# np.save('updated_adj_matrix.npy', updated_adj_matrix)
# print("Updated adjacency matrix saved to 'updated_adj_matrix.npy'")

# 加载保存的邻接矩阵
updated_adj_matrix = np.load('updated_adj_matrix.npy')
print("Loaded adjacency matrix from 'updated_adj_matrix.npy'")

# 创建图
# G = nx.from_numpy_array(updated_adj_matrix)

# # 设置节点名称
# unique_region_names = np.unique(region_names)
# mapping = {i: unique_region_names[i] for i in range(len(unique_region_names))}
# G = nx.relabel_nodes(G, mapping)

# # 打印调试信息
# print("Graph nodes after relabeling:", G.nodes())

# # 获取节点的三维坐标
# node_positions = {name: aggregated_spatial[i] for i, name in enumerate(unique_region_names)}

# # 打印调试信息
# print("Node positions:", node_positions)

# # 计算节点颜色
# cmap = plt.cm.get_cmap('viridis', len(unique_region_names))
# node_colors_dict = {name: cmap(i) for i, name in enumerate(unique_region_names)}

# # 绘制三维图
# fig = plt.figure(figsize=(24, 16))  # 增加图形尺寸
# ax = fig.add_subplot(111, projection='3d')

# # 调整视角距离
# ax.dist = 10

# # 绘制节点
# for node, (x, y, z) in node_positions.items():
#     color_value = node_colors_dict[node]
#     ax.scatter(x, y, z, color=color_value, s=4)  # 增加节点大小

# # 绘制边
# for edge in G.edges():
#     try:
#         x = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
#         y = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
#         z = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
#         ax.plot(x, y, z, color='#000000', linestyle='dotted', linewidth=0.1)  # 使用虚线和细线
#     except KeyError as e:
#         print(f"KeyError: {e} - Check if the node exists in node_positions")

# # 创建图例
# legend_elements = [Line2D([0], [0], marker='o', color='w', label=name,
#                         markerfacecolor=node_colors_dict[name], markersize=10)
#                 for name in unique_region_names]
# ax.legend(handles=legend_elements, title="Node Names", loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2)

# # 调整图像布局
# plt.subplots_adjust(right=0.8)  # 为图例留出空间

# # 设置标题和显示
# ax.set_title("3D Graph Structure with Updated Adjacency Matrix")
# plt.savefig('updated_graph_3d.png', bbox_inches='tight')
# plt.show()

# print("3D图结构已保存到 'updated_graph_3d.png'")

# 启用 bokeh 扩展
hv.extension('bokeh')

# 设置节点名称
unique_region_names = np.unique(region_names)

# 准备数据
links = []
max_weight = np.max(updated_adj_matrix)
min_weight = np.min(updated_adj_matrix[updated_adj_matrix > 0])  # 忽略零值
scale_factor = 0.1  # 缩放因子，用于调整线条宽度

for i in range(len(updated_adj_matrix)):
    for j in range(len(updated_adj_matrix)):
        if updated_adj_matrix[i][j] > 0:
            # 标准化权重并应用缩放因子
            normalized_weight = (updated_adj_matrix[i][j] - min_weight) / (max_weight - min_weight)
            scaled_weight = normalized_weight * scale_factor
            links.append((unique_region_names[i], unique_region_names[j], scaled_weight))

# 创建和弦图
chord = hv.Chord(links)
chord.opts(
    opts.Chord(
        labels='index',
        cmap='Category20',
        edge_color='lightgray',  # 使用更浅的灰色
        node_color='index',
        node_size=10,
        width=1000,
        height=1000,
        title="Chord Diagram of Node Connections",
        edge_alpha=0.3  # 增加透明度
    )
)

# 显示图形
hv.save(chord, 'chord_diagram.html')
print("Chord diagram saved to 'chord_diagram.html'")