import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"]
gra = np.loadtxt('gra.txt')
x = range(1, len(gra) + 1)
gra = -np.sort(-np.abs(gra))
end_index = int(len(x))
plt.plot(x[:end_index], gra[:end_index])
plt.xlabel('梯度下标', size=12)
plt.ylabel('排序后梯度绝对值大小', size=12)
plt.show()
