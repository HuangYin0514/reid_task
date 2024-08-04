import matplotlib.pyplot as plt
import numpy as np


# 定义函数
def func(x, mu=0, sigma=1):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# 生成 x 值
x = np.linspace(-2, 2, 400)
# 生成 y 值
y = func(x)


x1 = np.array([-1.2, -1.1, 0.8, 2])
y1 = func(x1, mu=np.mean(x1), sigma=np.std(x1))


# 绘制曲线
plt.figure(figsize=(8, 6))
plt.title("Curve of $func$")
plt.xlabel("x")
plt.ylabel("y")

plt.scatter(x1, y1, color="red", zorder=5, label="Points (x1, y1)")
plt.plot(x, y, label=r"$func$")

plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
