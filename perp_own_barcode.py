import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from treelib import Node, Tree

def create_persistent_barcode(image):
    thresholds = np.sort(np.unique(image))[::-1]
    components = {}
    barcode = []
    tree = Tree()
    tree.create_node("root", "root")

    for threshold in thresholds:
        binary = (image >= threshold).astype(int)
        labeled, num_features = label(binary)

        new_components = {}

        for i in range(1, num_features + 1):
            component_mask = (labeled == i)

            parent = None
            for old_id, old_mask in components.items():
                if np.any(component_mask & old_mask):
                    parent = old_id
                    break

            if parent is None:
                new_id = len(components) + len(new_components) + 1
                new_components[new_id] = component_mask
                barcode.append([new_id, threshold, None])
                tree.create_node(f"ID: {new_id}\nBirth: {threshold}", new_id, parent="root", data=component_mask)
            else:
                new_components[parent] = component_mask | components[parent]
                tree.get_node(parent).data = new_components[parent]

        for old_id in list(components.keys()):
            if old_id not in new_components:
                for bar in barcode:
                    if bar[0] == old_id and bar[2] is None:
                        bar[2] = threshold
                        tree.get_node(old_id).tag += f"\nDeath: {threshold}"
                        break

        components = new_components

    for bar in barcode:
        if bar[2] is None:
            bar[2] = 0
            tree.get_node(bar[0]).tag += "\nDeath: 0"

    return barcode, tree

def plot_tree_with_components(tree, image):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('off')

    def plot_node(node, pos, level=0, dx=1.0):
        print(f"Debug: node = {node.identifier}, pos = {pos}")  # デバッグ出力
        if node.is_root():
            x, y = pos.get(node.identifier, (0, 0))  # デフォルト値を設定
        else:
            parent_pos = pos.get(node.bpointer, (0, 0))  # 親ノードの位置を取得（デフォルト値あり）
            x, y = parent_pos[0] + dx * (1 if node.identifier % 2 == 0 else -1), parent_pos[1] - 1

        pos[node.identifier] = (x, y)  # 現在のノードの位置を更新
        ax.text(x, y, node.tag, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))

        if node.data is not None:
            component_image = np.zeros_like(image)
            component_image[node.data] = 255
            extent = (x - 0.5, x + 0.5, y - 0.5, y + 0.5)
            ax.imshow(component_image, cmap='gray', extent=extent, alpha=0.5)

        for child in tree.children(node.identifier):
            plot_node(child, pos, level + 1, dx / 2)
            ax.plot([x, pos[child.identifier][0]], [y, pos[child.identifier][1]], 'k-')

    plot_node(tree.get_node("root"), {tree.get_node("root").identifier: (0, 0)}, 0, 1.0)
    plt.savefig('tree.png')

# Input image
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

# Create persistent barcode and tree
barcode, tree = create_persistent_barcode(image)

# Print the barcode
for bar in barcode:
    print(f"Component {bar[0]}: Birth = {bar[1]}, Death = {bar[2]}")

# Plot the tree with component images
plot_tree_with_components(tree, image)

# Visualize the barcode
plt.figure(figsize=(10, 6))
for bar in barcode:
    plt.plot([bar[1], bar[2]], [bar[0], bar[0]], 'b-', linewidth=2)

plt.xlabel('Threshold')
plt.ylabel('Component ID')
plt.title('Persistent Barcode')
plt.gca().invert_yaxis()
plt.savefig('barcode.png')
