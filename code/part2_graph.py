import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（解决图表中文乱码）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取你的实验结果
df = pd.read_csv("hallucination_experiment_result.csv")

# ======================
# 计算指标
# ======================
acc_qwen = df["qwen7b_correct"].mean()
acc_grok = df["grok3mini_correct"].mean()
hallu_qwen = 1 - acc_qwen
hallu_grok = 1 - acc_grok

correct_qwen = df["qwen7b_correct"].sum()
correct_grok = df["grok3mini_correct"].sum()
total = len(df)

models = ["Qwen2.5-7B", "Grok 3 Mini"]
accs = [acc_qwen, acc_grok]
hallus = [hallu_qwen, hallu_grok]

# ======================
# 图1：幻觉率对比
# ======================
plt.figure(figsize=(7, 5))
plt.bar(models, hallus, color=["#4472c4", "#ed7d31"], width=0.6)
plt.title("Model hallucination rate comparison", fontsize=14)
plt.ylabel("Hallucination rate", fontsize=12)
plt.ylim(0, 0.6)
for i, v in enumerate(hallus):
    plt.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("hallucination_compare.png", dpi=300)

# ======================
# 图2：准确率对比
# ======================
plt.figure(figsize=(7, 5))
plt.bar(models, accs, color=["#5cb85c", "#f0ad4e"], width=0.6)
plt.title("Model accuracy comparison", fontsize=14)
plt.ylabel("accuracy", fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(accs):
    plt.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("accuracy_compare.png", dpi=300)

# ======================
# 图3：正确/错误数量堆叠对比
# ======================
correct_counts = [correct_qwen, correct_grok]
wrong_counts = [total - correct_qwen, total - correct_grok]

x = np.arange(len(models))
width = 0.6

plt.figure(figsize=(8, 5))
plt.bar(x, correct_counts, width, label="True", color="#5cb85c")
plt.bar(x, wrong_counts, width, bottom=correct_counts, label="False", color="#d9534f")

plt.title("Qwen vs Grok Comparison of correct/incorrect numbers", fontsize=14)
plt.ylabel("Sample size", fontsize=12)
plt.xticks(x, models)
plt.legend()
plt.ylim(0, total)

for i in range(2):
    plt.text(i, correct_counts[i] / 2, str(correct_counts[i]), ha="center", color="white", fontweight=600)
    plt.text(i, correct_counts[i] + wrong_counts[i] / 2, str(wrong_counts[i]), ha="center", color="white", fontweight=600)

plt.tight_layout()
plt.savefig("correct_wrong_distribute.png", dpi=300)

print(" The chart has been generated")
print(" Hallucination rate：Qwen %.2f%% | Grok %.2f%%" % (hallu_qwen*100, hallu_grok*100))
print(" Accuracy：Qwen %.2f%% | Grok %.2f%%" % (acc_qwen*100, acc_grok*100))