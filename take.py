import higra as hg
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def find_max_altitude_leaf(tree, altitudes, node):
    if tree.num_children(node) == 0:  # 葉ノード
        return node, altitudes[node]

    children = list(tree.children(node))
    max_leaf = None
    max_altitude = float('-inf')

    for child in children:
        leaf, altitude = find_max_altitude_leaf(tree, altitudes, child)
        if altitude > max_altitude:
            max_leaf = leaf
            max_altitude = altitude

    return max_leaf, max_altitude

def create_persistent_barcode(tree, altitudes):
    n_vertices = tree.num_vertices()
    barcode = []
    active_components = {}  # key: node, value: (birth_altitude, node)

    # ノードをaltitudeの昇順にソート (葉ノードからルートノードに向かって処理)
    leaves = set(tree.leaves())
    print("n_vertices", range(n_vertices))
    sorted_nodes = sorted(range(n_vertices), key=lambda x: altitudes[x], reverse=True)
    print("sorted_nodes", sorted_nodes)

    for node in sorted_nodes:
        current_altitude = altitudes[node]
        parent = tree.parent(node)

        if tree.num_children(node) == 0:  # 葉ノード
            active_components[node] = (current_altitude, node)
        else:  # 内部ノード
            children = list(tree.children(node))

            if len(children) > 1:
                # 子ノードの中で最も高いaltitudeを持つものを見つける
                max_child, _ = max((find_max_altitude_leaf(tree, altitudes, child) for child in children), key=lambda x: x[1])
                print("max_child", max_child)

                for child in children:
                    if child in active_components:
                        birth_altitude, birth_node = active_components[child]
                        if child != max_child:
                            # 最大の子以外の成分は消滅
                            barcode.append((birth_altitude, current_altitude))
                        else:
                            # 最大の子の成分は継続
                            active_components[node] = (birth_altitude, birth_node)
                        del active_components[child]
            else:
                # 子ノードが1つの場合、その成分を継続
                child = children[0]
                if child in active_components:
                    active_components[node] = active_components[child]
                    del active_components[child]

    # ルートノードの処理
    root = tree.root()
    if root in active_components:
        birth_altitude, _ = active_components[root]
        barcode.append((birth_altitude, 1.0))

    return barcode

image = sitk.GetArrayFromImage(sitk.ReadImage("1.mhd"))
image = (image - image.min()) / (image.max() - image.min())
# image = np.array([
#     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
#     [0.3, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.7, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.9, 0.9, 0.7, 0.7, 0.6, 0.6, 0.3],
#     [0.3, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.3],
#     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# ])
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('image.png')


def save_tree(tree, altitudes, max_or_min="max"):
    leaves = set(tree.leaves())
    deleted_vertices = np.zeros(tree.num_vertices(), dtype=bool)
    deleted_vertices[list(leaves)] = True
    new_tree, node_map = hg.simplify_tree(tree, deleted_vertices, process_leaves=True)
    new_altitudes = altitudes[node_map]

    def plot_tree(tree, altitudes, ax):
        positions = {}
        for i in range(tree.num_vertices()):
            positions[i] = (i, altitudes[i])

        for i in range(tree.num_vertices()):
            parent = tree.parents()[i]
            if parent != i:
                ax.plot([positions[i][0], positions[parent][0]], [positions[i][1], positions[parent][1]], 'k-')

        for i in range(tree.num_vertices()):
            ax.text(positions[i][0], positions[i][1], str(i), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(new_tree, new_altitudes, ax)
    ax.set_title(f"{max_or_min.capitalize()}-tree")
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Altitude')
    ax.set_ylim(1, 0)
    ax.grid(True)

    # 画像として保存
    plt.savefig(f'{max_or_min}_tree.png')

graph = hg.get_4_adjacency_graph(image.shape)
# Max-treeの構築
max_tree, max_altitudes = hg.component_tree_max_tree(graph, image.flatten())
min_tree, min_altitudes = hg.component_tree_min_tree(graph, image.flatten())

save_tree(max_tree, max_altitudes, "max")
save_tree(min_tree, min_altitudes, "min")