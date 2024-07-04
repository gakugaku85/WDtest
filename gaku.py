import higra as hg
import numpy as np
import matplotlib.pyplot as plt

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
                            barcode.append((1.0-birth_altitude, 1.0-current_altitude))
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
        barcode.append((1.0-active_components[root], 1.0))

    return barcode

# サンプル画像の定義と正規化
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 1, 1, 1, 1, 0],
    [0, 5, 5, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 5, 5, 0],
    [0, 1, 1, 0, 0, 5, 5, 0],
    [0, 7, 7, 2, 2, 5, 5, 0],
    [0, 7, 7, 2, 2, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.float32)
image = image / 10

# グラフの作成
graph = hg.get_4_adjacency_graph(image.shape)

# Max-treeの構築
tree, altitudes = hg.component_tree_max_tree(graph, image.flatten())

print("altitudes", altitudes)

leaves = set(tree.leaves())

deleted_vertices = np.zeros(tree.num_vertices(), dtype=bool)
deleted_vertices[list(leaves)] = True

new_tree, node_map = hg.simplify_tree(tree, deleted_vertices, process_leaves=True)

new_altitudes = altitudes[node_map]

print("Original tree:", tree.parents())
print("newtree_leaf", set(new_tree.leaves()))
print("New tree:", new_tree.parents())
print("New altitudes:", new_altitudes)
print("Node map:", node_map)
print("root", new_tree.root())

print("New tree structure:")
# print("New tree shape:", new_tree.shape())
print("New altitudes shape:", new_altitudes.shape)

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
ax.set_title('Simplified Tree')
ax.set_xlabel('Node Index')
ax.set_ylabel('Altitude')
# y軸のrangeを1から0に変更
ax.set_ylim(1, 0)
ax.grid(True)

# 画像として保存
plt.savefig('simplified_tree.png')

# Persistent barcodeの作成
barcode = create_persistent_barcode(new_tree, new_altitudes)

# Barcodeの表示
fig, ax = plt.subplots(figsize=(10, 6))

# バーを長さ順にソート
sorted_barcode = sorted(barcode, key=lambda x: x[1] - x[0])

for i, (birth, death) in enumerate(sorted_barcode):
    if birth != death:
        print(f"Birth: {birth:.2f}, Death: {death:.2f}")

for i, (birth, death) in enumerate(sorted_barcode):
    ax.barh(i, death - birth, left=birth, height=0.6, color='blue', alpha=0.6)

ax.set_xlabel('1 - Threshold')
ax.set_title('Persistent Barcode')
ax.set_ylim(-1, len(barcode))
ax.set_xlim(0, 1)

# Y軸のラベルを非表示にする
ax.set_yticks([])

plt.tight_layout()
plt.savefig('barcode.png')

# 連結成分の数を表示
print(f"Number of connected components: {len(barcode)}")
print("Barcode: ", barcode)
