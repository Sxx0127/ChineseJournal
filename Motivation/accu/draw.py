import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

quan = []
with open('quan.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        quan.append(float(line[line.find('y') + 3:]))
print(quan)

top = []
with open('top.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top.append(float(line[line.find('y') + 3:]))
print(top)

top_quan = []
with open('top_quan.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top_quan.append(float(line[line.find('y') + 3:]))
print(top_quan)

top_diff_quan = []
with open('top_diff_quan.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top_diff_quan.append(float(line[line.find('y') + 3:]))
print(top_diff_quan)

start_index = 0
end_index = 20
#  一个客户端一轮迭代0.01MB，一次十个客户端，每5次输出一个结果，所以以0.5为间隔
top_diff_quan_x = np.arange(0.5, 10.5, 0.5)
plt.plot(top_diff_quan_x, top_diff_quan[:len(top_diff_quan_x)], marker='o', label="Top 1% with different QSGD", markevery=2)
# x = [i for i in range(5, 105, 5)]
# x = x[start_index:end_index]
# quan = quan[start_index:end_index]
# top = top[start_index:end_index]
# top_quan = top_quan[start_index:end_index]
# top_diff_quan = top_diff_quan[start_index:end_index]

#  一个客户端一轮迭代0.01MB，一次十个客户端，每5次输出一个结果，所以以0.5为间隔
top_quan_x = np.arange(0.5, 10.5, 0.5)
plt.plot(top_quan_x, top_quan[:len(top_quan_x)], marker='+', label="Top 1% with $QSGD_{6}$", markevery=2)

#  一个客户端一轮迭代0.02MB，一次十个客户端，每5次输出一个结果，所以以1为间隔
top_x = np.arange(1, 10.5, 1)
plt.plot(top_x, top[:len(top_x)], marker='D', label="Top 1%", markevery=2)

#  一个客户端一轮迭代0.2MB，一次十个客户端，每5次输出一个结果，所以以10为间隔
quan_x = np.arange(10, 10.5, 10)
plt.plot(quan_x, quan[:len(quan_x)], marker='*', label="$QSGD_{6}$", markevery=2)
plt.xlabel('Uploaded Traffic (MB)')
plt.ylabel('Test Accuracy')
# plt.ylim(30)
# plt.title('ResNet-18')
plt.legend(loc='best')
plt.show()
