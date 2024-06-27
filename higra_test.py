import higra as hg
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# image = np.array([
#     [8, 8, 8, 8, 8, 8, 8, 8],
#     [8, 2, 2, 2, 2, 2, 2, 8],
#     [8, 2, 5, 5, 5, 5, 2, 8],
#     [8, 2, 5, 0, 0, 5, 2, 8],
#     [8, 2, 5, 0, 0, 5, 2, 8],
#     [8, 2, 5, 5, 5, 5, 2, 8],
#     [8, 2, 2, 2, 2, 2, 2, 8],
#     [8, 8, 8, 8, 8, 8, 8, 8]
# ])

image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 0],
    [0, 0, 0, 0, 0, 5, 5, 0],
    [0, 7, 7, 2, 2, 5, 5, 0],
    [0, 7, 7, 2, 2, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

def print_tree(node, level=0):
    print(" " * level + f"{altitudes[node]}")
    for child in tree.children(node):
        print_tree(child, level + 2)

image = image / 10

plt.imshow(image, cmap='gray')
plt.savefig('image.png')
plt.close()

# コンポーネントツリーの作成
graph = hg.get_8_adjacency_graph(image.shape)

hg.plot_graph(graph, vertex_positions=np.array([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])]))
plt.savefig('graph.png')

tree, altitudes = hg.component_tree_max_tree(graph, image)

print_tree(tree.root())

print("root", tree.root())
print("num_leaves", tree.num_leaves())
print("num_vertices", tree.num_vertices())

print("parent", tree.parents())

print("max_tree")

# hg.plot_partition_tree(tree, altitudes=altitudes)
# plt.savefig('dendrogram.png')
# hg.print_partition_tree(tree, altitudes=altitudes)
# plt.close()

tree, altitudes = hg.component_tree_min_tree(graph, image)

print("min_tree")

# hg.plot_partition_tree(tree, altitudes=altitudes)
# plt.savefig('dendrogram2.png')
# hg.print_partition_tree(tree, altitudes=altitudes)
# plt.close()

# ツリーの簡単な情報表示
print(f'Number of nodes in the tree: {tree.num_vertices()}')
print(f'Number of leaves in the tree: {tree.num_leaves()}')