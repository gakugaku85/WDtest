import higra as hg
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import gudhi as gd
import time

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
    active_components = {}  # key: node, value: birth_altitude

    # ノードをaltitudeの降順にソート (ルートノードから葉ノードに向かって処理)
    sorted_nodes = sorted(range(n_vertices), key=lambda x: altitudes[x], reverse=True)

    for node in sorted_nodes:
        current_altitude = altitudes[node]
        children = list(tree.children(node))

        if not children:  # 葉ノード
            active_components[node] = current_altitude
        else:  # 内部ノード
            child_components = [active_components[child] for child in children if child in active_components]

            if child_components:
                # 子ノードの中で最も高いaltitudeを持つ葉ノードを見つける
                max_child, _ = max((find_max_altitude_leaf(tree, altitudes, child) for child in children), key=lambda x: x[1])

                for child in children:
                    if child in active_components:
                        birth_altitude = active_components[child]
                        leaf, _ = find_max_altitude_leaf(tree, altitudes, child)
                        if leaf != max_child:
                            # 最大のaltitudeを持つ葉ノード以外の成分は消滅
                            barcode.append((1.0 - birth_altitude, 1.0 - current_altitude))
                        else:
                            # 最大のaltitudeを持つ葉ノードの成分は継続
                            active_components[node] = birth_altitude
                        del active_components[child]
            else:
                # 新しい連結成分が生まれる場合
                active_components[node] = current_altitude

    # ルートノードの処理
    root = tree.root()
    if root in active_components:
        barcode.append((1.0 - active_components[root], 1.0))

    return barcode

def create_min_persistent_barcode(tree, altitudes):
    n_vertices = tree.num_vertices()
    barcode = []
    active_components = {}  # key: node, value: birth_altitude

    # ノードをaltitudeの昇順にソート (葉ノードからルートノードに向かって処理)
    sorted_nodes = sorted(range(n_vertices), key=lambda x: altitudes[x])

    for node in sorted_nodes:
        current_altitude = altitudes[node]
        children = list(tree.children(node))

        if not children:  # 葉ノード
            active_components[node] = current_altitude
        else:  # 内部ノード
            child_components = [active_components[child] for child in children if child in active_components]

            if child_components:
                # 子ノードの中で最も低いaltitudeを持つ葉ノードを見つける
                min_child, _ = min((find_max_altitude_leaf(tree, altitudes, child) for child in children), key=lambda x: x[1])

                for child in children:
                    if child in active_components:
                        birth_altitude = active_components[child]
                        leaf, _ = find_max_altitude_leaf(tree, altitudes, child)
                        if leaf != min_child:
                            # 最小のaltitudeを持つ葉ノード以外の成分は消滅
                            barcode.append((1.0 - birth_altitude, 1.0 - current_altitude))
                        else:
                            # 最小のaltitudeを持つ葉ノードの成分は継続
                            active_components[node] = birth_altitude
                        del active_components[child]
            else:
                # 新しい連結成分が生まれる場合
                active_components[node] = current_altitude

    # ルートノードの処理
    root = tree.root()
    if root in active_components:
        barcode.append((1.0 - active_components[root], 1.0))

    return barcode

# サンプル画像の定義と正規化

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
# image = image / 10

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('image.png')

# グラフの作成
start_time = time.time()
graph = hg.get_4_adjacency_graph(image.shape)

max_tree, max_altitudes = hg.component_tree_max_tree(graph, image.flatten())

min_tree, min_altitudes = hg.component_tree_min_tree(graph, image.flatten())

# Persistent barcodeの作成 (Max-tree)
max_barcode = create_persistent_barcode(max_tree, max_altitudes)

# Persistent barcodeの作成 (Min-tree)
min_barcode = create_min_persistent_barcode(min_tree, min_altitudes)

# max treeとmin treeの配列を結合し、persistenceを作成
persistence = []
for birth, death in max_barcode:
    if birth != death:
        persistence.append((0, (birth, death)))
for birth, death in min_barcode:
    if birth != death:
        persistence.append((1, (birth, death)))

# birthの値でソート
persistence = sorted(persistence, key=lambda x: x[1][0])

cc_num = 0
hole_num = 0
for idx, (birth, death) in persistence:
    if idx == 0:
        cc_num += 1
    else:
        hole_num += 1

print("max min tree :cc_num", cc_num, "hole num", hole_num)
end_time = time.time()

print("time", end_time - start_time)

gd.plot_persistence_barcode(persistence)
plt.xlim(0, 1)
plt.title('Persistent Barcode (Max-tree and Min-tree)')
plt.savefig('combined_barcode.png')


gudhi_time = time.time()
cc = gd.CubicalComplex(
        dimensions=image.shape, top_dimensional_cells=1-image.flatten()
    )
persistence = cc.persistence()

gudhi_end_time = time.time()
print("gudhi time", gudhi_end_time - gudhi_time)

gd.plot_persistence_barcode(persistence)
plt.xlim(0, 1)
plt.savefig('gudhi_barcode.png')
cc_num = 0
hole_num = 0
for idx, (birth, death) in persistence:
    if idx == 0:
        cc_num += 1
    else:
        hole_num += 1
print("gudhi:cc_num", cc_num, "hole num", hole_num)

