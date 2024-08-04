import matplotlib.pyplot as plt
import numpy as np


# 定义函数
def func(x, mu=0, sigma=1):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# 生成 x 值
x = np.linspace(-2, 2, 400)
# 生成 y 值
y = func(x)

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$func$")
plt.title("Curve of $func$")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
