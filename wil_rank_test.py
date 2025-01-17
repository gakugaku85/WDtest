import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_rel, wilcoxon

# データの読み込み
file_path = '../SR3/SR3_wdTest/experiments/s14_best_cc_kentei.xlsx'  # ファイルパスを指定
data = pd.read_excel(file_path)

# 検定を実行する関数
def perform_test(val1, val2):
    # 正規性検定
    # if shapiro(val1).pvalue > 0.05 and shapiro(val2).pvalue > 0.05:
    #     # 両方が正規分布に従う場合：対応のある t 検定
    #     # return ttest_rel(val1, val2)
    # else:
        # 非正規分布の場合：ウィルコクソン符号付順位検定
    return wilcoxon(val1, val2)

# 各列の検定結果
test_result_val1 = perform_test(data["ori_val1"], data["wd_val1"])
test_result_val2 = perform_test(data["ori_val2"], data["wd_val2"])

# データを可視化用に整形
data_melted = pd.melt(
    data,
    id_vars=["fname"],
    value_vars=["ori_val1", "wd_val1", "ori_val2", "wd_val2"],
    var_name="Category",
    value_name="Value"
)

# 箱ひげ図の作成
plt.figure(figsize=(10, 6))
sns.boxplot(x="Category", y="Value", data=data_melted)
categories = ["ori_val1", "wd_val1", "ori_val2", "wd_val2"]
means = data_melted.groupby("Category")["Value"].mean().reindex(categories)
print(means)
for i, mean in enumerate(means):
    plt.scatter(i, mean, color="red", label="Mean" if i == 0 else "", zorder=5)

# 有意差を示すアスタリスクの追加
x_coords = [0.5, 2.5]  # 各グループの中心位置
y_coords = [max(data["ori_val1"].max(), data["wd_val1"].max()) + 0.5,
            max(data["ori_val2"].max(), data["wd_val2"].max()) + 0.5]  # アスタリスクのY座標

if test_result_val1.pvalue < 0.05:
    plt.text(x_coords[0], y_coords[0], '*', ha='center', va='bottom', fontsize=20, color='red')

if test_result_val2.pvalue < 0.05:
    plt.text(x_coords[1], y_coords[1], '*', ha='center', va='bottom', fontsize=20, color='red')

# グラフの調整
plt.title("Boxplot with Significance Testing", fontsize=14)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.tight_layout()
plt.savefig("../SR3/SR3_wdTest/experiments/s14_best_cc_kentei.png")

# 検定結果の出力
print("ori_val1 vs wd_val1:", test_result_val1)
print("ori_val2 vs wd_val2:", test_result_val2)
