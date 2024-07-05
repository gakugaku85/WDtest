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

# image = image = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 5, 5, 1, 1, 1, 1, 0],
#     [0, 5, 5, 1, 1, 1, 1, 0],
#     [0, 1, 1, 0, 0, 5, 5, 0],
#     [0, 1, 1, 0, 0, 5, 5, 0],
#     [0, 7, 7, 2, 2, 5, 5, 0],
#     [0, 7, 7, 2, 2, 5, 5, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=np.float32)
# image = image / 10

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('image.png')

# グラフの作成
graph = hg.get_4_adjacency_graph(image.shape)

# Max-treeの構築
max_tree, max_altitudes = hg.component_tree_max_tree(graph, image.flatten())

# print("num_parents", len(max_tree.parents()))
# 閾値のリストを作成 (画像の画素値ごと)
thresholds = np.unique(image)[1:]

# 閾値ごとに画像を表示
fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 7))
axes = axes.ravel()

for i, threshold in enumerate(thresholds):
    filtered = hg.reconstruct_leaf_data(max_tree, max_altitudes >= threshold)
    filtered = filtered.reshape(image.shape)

    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(f'Threshold: {threshold:.2f}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('max_thresholds.png')

# Min-treeの構築
min_tree, min_altitudes = hg.component_tree_min_tree(graph, image.flatten())

# print("num_parents", len(min_tree.parents()))

thresholds = np.unique(image)[1:]

# 閾値ごとに画像を表示
fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 7))
axes = axes.ravel()

for i, threshold in enumerate(thresholds):
    filtered = hg.reconstruct_leaf_data(min_tree, min_altitudes < threshold)
    filtered = filtered.reshape(image.shape)

    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(f'Threshold: {threshold:.2f}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('min_thresholds.png')

# Persistent barcodeの作成 (Max-tree)
max_barcode = create_persistent_barcode(max_tree, max_altitudes)

# Persistent barcodeの作成 (Min-tree)
min_barcode = create_min_persistent_barcode(min_tree, min_altitudes)

# Max-treeとMin-treeのBarcodeの表示
fig, ax = plt.subplots(figsize=(10, 6))

# Max-treeのバーを長さ順にソート
sorted_max_barcode = sorted(max_barcode, key=lambda x: x[1] - x[0])
sorted_min_barcode = sorted(min_barcode, key=lambda x: x[1] - x[0])

i=0
for birth, death in sorted_min_barcode:
    if birth != death:
        i+=1
        ax.barh(i, death - birth, left=birth, height=0.6, color='red', alpha=0.6 ,label='Max-tree' if i == 0 else "")

j=0
for birth, death in sorted_max_barcode:
    if birth != death:
        j+=1
        ax.barh(j + i, death - birth, left=birth, height=0.6, color='blue', alpha=0.6, label='Min-tree' if j == 0 else "")

ax.set_xlabel('1 - Threshold')
ax.set_title('Persistent Barcode (Max-tree and Min-tree)')
ax.set_ylim(-1 , i + j + 1)
ax.set_xlim(0, 1)
ax.grid(axis='x', linestyle='--', alpha=0.6,)

# Y軸のラベルを非表示にする
ax.set_yticks([])

# 凡例を追加
ax.legend()

plt.tight_layout()
plt.savefig('combined_barcode.png')

# 連結成分の数を表示
print("tree:cc_num", j, "hole num", i)


import gudhi as gd

cc = gd.CubicalComplex(
        dimensions=image.shape, top_dimensional_cells=1-image.flatten()
    )
persistence = cc.persistence()
cc_num = 0
hole_num = 0
for idx, (birth, death) in persistence:
    if idx == 0:
        cc_num += 1
    else:
        hole_num += 1
print("gudhi:cc_num", cc_num, "hole num", hole_num)

