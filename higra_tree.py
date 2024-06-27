import higra as hg
import numpy as np
import matplotlib.pyplot as plt

# 画像データの作成
image = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 2, 2, 2, 2, 2, 2, 8],
    [8, 2, 5, 5, 5, 5, 2, 8],
    [8, 2, 5, 9, 9, 5, 2, 8],
    [8, 2, 5, 9, 9, 5, 2, 8],
    [8, 2, 5, 5, 5, 5, 2, 8],
    [8, 2, 2, 2, 2, 2, 2, 8],
    [8, 8, 8, 8, 8, 8, 8, 8]
])

# 画像の表示
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.savefig('image.png')
plt.close()

# コンポーネントツリーの作成
graph = hg.get_4_adjacency_graph(image.shape)
tree, altitudes = hg.component_tree_max_tree(graph, image)
area = hg.attribute_area(tree)

result = hg.reconstruct_leaf_data(tree, altitudes, area < 100)
plt.imshow(result, cmap='gray')
plt.savefig('reconstructed_image.png')

print(area)
print(altitudes)

# ツリーの簡単な情報表示
print(f'Number of nodes in the tree: {tree.num_vertices()}')
print(f'Number of leaves in the tree: {tree.num_leaves()}')

# 各ノードに対応する二値化画像を生成
binary_images = []
num_leaves = tree.num_leaves()

# 各ノードの領域を二値化画像として再構築
for node in range(tree.num_vertices()):
    deleted_nodes = np.ones(tree.num_vertices(), dtype=bool)
    deleted_nodes[node] = False
    mask = hg.reconstruct_leaf_data(tree, deleted_nodes).reshape(image.shape)
    binary_images.append(mask)

# 画像の数に応じてグリッドのサイズを計算
num_images = len(binary_images)
grid_size = int(np.ceil(np.sqrt(num_images)))

# グリッドに画像を並べて表示
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        ax.imshow(binary_images[i], cmap='gray')
        ax.set_title(f'Node {i}')
    ax.axis('off')

vertex_positions = np.array([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])])
vertex_labels = np.random.randint(0, 10, size=graph.num_vertices())

plt.tight_layout()
plt.savefig('binary_images_grid.png')
plt.close()

# コンポーネントツリーのプロット
fig, ax = plt.subplots(figsize=(10, 10))
# hg.plot_partition_tree(tree)
hg.plot_graph(graph, vertex_positions=vertex_positions, vertex_labels=vertex_labels)

plt.savefig('component_tree_with_labels.png')
plt.close()
