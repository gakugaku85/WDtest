import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import networkx as nx

# 画像の読み込み
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 1, 1, 1, 1, 0],
    [0, 5, 5, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 5, 5, 0],
    [0, 1, 1, 0, 0, 5, 5, 0],
    [0, 7, 7, 2, 2, 5, 5, 0],
    [0, 7, 7, 2, 2, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

# グラフの作成
graph = hg.get_4_adjacency_graph(image.shape)

# Max-treeの構築
tree, altitudes = hg.component_tree_min_tree(graph, image.flatten())

# 閾値のリストを作成 (画像の画素値ごと)
thresholds = np.unique(image)

# 閾値ごとに画像を表示
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes = axes.ravel()

for i, threshold in enumerate(thresholds):  # 最初の10個の閾値のみ表示
    filtered = hg.reconstruct_leaf_data(tree, altitudes > threshold)
    filtered = filtered.reshape(image.shape)

    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(f'Threshold: {threshold:.2f}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('thresholds.png')

# Max-treeの視覚化（葉ノードを除外）
def plot_hierarchical_max_tree(tree, altitudes):
    G = nx.DiGraph()
    leaves = set(tree.leaves())
    for i in range(tree.num_vertices()):
        if i not in leaves:
            G.add_node(i, altitude=altitudes[i])
            parent = tree.parent(i)
            if parent != i and parent not in leaves:
                G.add_edge(i, parent)  # エッジの方向を逆にする

    # 階層的レイアウトを使用
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_size=100, node_color='lightblue', with_labels=False,
            arrows=True, arrowsize=20, edge_color='gray')

    # ノードのラベルを表示
    labels = {node: f"{node}\n{G.nodes[node]['altitude']:.2f}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Hierarchical Max-tree Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('max_tree.png')

# 葉ノードを除外したMax-treeの視覚化を実行
plot_hierarchical_max_tree(tree, altitudes)