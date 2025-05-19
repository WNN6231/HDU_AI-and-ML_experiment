import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

A = np.array([1, 1, 0.5, 0.2, 0.0])

R = np.array([
    [0.5, 0.5, 0.5, 0.8, 1.0],
    [0.5, 0.5, 0.5, 0.8, 1.0],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])

def fuzzy_inference_max_min(A, R):
    result = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        min_vals = np.minimum(A, R[:, j])
        result[j] = np.max(min_vals)
    return result

def fuzzy_inference_max_prod(A, R):
    result = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        prod_vals = A * R[:, j]
        result[j] = np.max(prod_vals)
    return result

def defuzzification_centroid(B):
    x = np.arange(1, len(B) + 1)
    return np.sum(B * x) / np.sum(B) if np.sum(B) != 0 else 0

B_min = fuzzy_inference_max_min(A, R)
B_prod = fuzzy_inference_max_prod(A, R)

output_min = defuzzification_centroid(B_min)
output_prod = defuzzification_centroid(B_prod)

print("Max-Min 合成结果:", B_min)
print("Max-Prod 合成结果:", B_prod)
print(f"重心法去模糊化结果 (Max-Min): {output_min:.2f}")
print(f"重心法去模糊化结果 (Max-Prod): {output_prod:.2f}")

x = np.arange(1, len(A) + 1)
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
marker1, stemlines1, baseline1 = plt.stem(x, A)
plt.setp(marker1, marker='o', markersize=8)
plt.title("输入模糊集合 A'")
plt.xlabel("元素索引")
plt.ylabel("隶属度")

plt.subplot(1, 3, 2)
marker2, stemlines2, baseline2 = plt.stem(x, B_min)
plt.setp(marker2, marker='s', markersize=8)
plt.title("输出模糊集合 B (Max-Min)")
plt.xlabel("元素索引")

plt.subplot(1, 3, 3)
marker3, stemlines3, baseline3 = plt.stem(x, B_prod)
plt.setp(marker3, marker='^', markersize=8)
plt.title("输出模糊集合 B (Max-Prod)")
plt.xlabel("元素索引")

plt.tight_layout()
plt.show()
