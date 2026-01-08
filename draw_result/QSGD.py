import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("TkAgg")

x = [i for i in range(5, 105, 5)]

IID_quan12 = []
with open('../CIFAR_IID_QSGD_result/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan12.append(float(line[line.find('y') + 3:]))
# print(IID_quan12)

IID_quan8 = []
with open('../CIFAR_IID_QSGD_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan8.append(float(line[line.find('y') + 3:]))
# print(IID_quan8)

IID_quan10 = []
with open('../CIFAR_IID_QSGD_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan10.append(float(line[line.find('y') + 3:]))
# print(IID_quan10)

IID_quan32 = []
with open('../CIFAR_IID_QSGD_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan32.append(float(line[line.find('y') + 3:]))
# print(IID_quan32)

IID_work = []
with open('../CIFAR_IID_QSGD_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_work.append(float(line[line.find('y') + 3:]))
# print(IID_work)
end_index = 16
x = x[:end_index]
IID_quan12 = IID_quan12[:end_index]
IID_quan8 = IID_quan8[:end_index]
IID_quan10 = IID_quan10[:end_index]
IID_quan32 = IID_quan32[:end_index]
IID_work = IID_work[:end_index]
base = max(max(IID_quan12), max(IID_quan8), max(IID_quan10), max(IID_quan32))
print("Accu improve ", (max(IID_work) - base) / base)
fig = plt.figure(figsize=(10, 8))
plt.subplot(231)
plt.xlabel("全局迭代", size=13, fontproperties="SimSun")
plt.ylabel("模型性能(%)", size=13, fontproperties="SimSun")
plt.plot(x, IID_quan8, marker='D', label="$Q_8Top_k$", markevery=2)
plt.plot(x, IID_quan10, marker='+', label="$Q_{10}Top_k$", markevery=2)
plt.plot(x, IID_quan12, marker='*', label="$Q_{12}Top_k$", markevery=2)
plt.plot(x, IID_quan32, marker='o', label="$Top_k$", markevery=2)
plt.plot(x, IID_work, marker='^', label="Fed-CVLC", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, ncol=5, prop={'size': 13})
plt.title('(a)CIFAR-10_QSGD_IID', y=-0.3)

target_acc = min(max(IID_quan8), max(IID_quan10), max(IID_quan12), max(IID_quan32), max(IID_work))
traffic_8 = (np.where(np.array(IID_quan8) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_10 = (np.where(np.array(IID_quan10) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_12 = (np.where(np.array(IID_quan12) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_32 = (np.where(np.array(IID_quan32) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_work = (np.where(np.array(IID_work) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
print("CIFAR10_IID, Accu: {}, quan8: {}, quan10: {}, quan12: {}, quan32: {}, work: {}"
      .format(target_acc, traffic_8, traffic_10, traffic_12, traffic_32, traffic_work))

NIID_quan12 = []
with open('../CIFAR_NIID_QSGD_result/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan12.append(float(line[line.find('y') + 3:]))
# print(NIID_quan12)

NIID_quan8 = []
with open('../CIFAR_NIID_QSGD_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan8.append(float(line[line.find('y') + 3:]))
# print(NIID_quan8)

NIID_quan10 = []
with open('../CIFAR_NIID_QSGD_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan10.append(float(line[line.find('y') + 3:]))
# print(NIID_quan10)

NIID_quan32 = []
with open('../CIFAR_NIID_QSGD_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan32.append(float(line[line.find('y') + 3:]))
# print(NIID_quan32)

NIID_work = []
with open('../CIFAR_NIID_QSGD_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_work.append(float(line[line.find('y') + 3:]))
# print(NIID_work)

NIID_quan12 = NIID_quan12[:end_index]
NIID_quan8 = NIID_quan8[:end_index]
NIID_quan10 = NIID_quan10[:end_index]
NIID_quan32 = NIID_quan32[:end_index]
NIID_work = NIID_work[:end_index]
base = max(max(NIID_quan12), max(NIID_quan8), max(NIID_quan10), max(NIID_quan32))
print("Accu improve ", (max(NIID_work) - base) / base)
# fig = plt.figure(figsize=(5, 3))
plt.subplot(232)
plt.xlabel("全局迭代", size=13, fontproperties="SimSun")
plt.ylabel("模型性能(%)", size=13, fontproperties="SimSun")
plt.plot(x, NIID_quan12, marker='*', label="Spar. with $QSGD_{12}$", markevery=2)
plt.plot(x, NIID_quan8, marker='D', label="Spar. with $QSGD_8$", markevery=2)
plt.plot(x, NIID_quan10, marker='+', label="Spar. with $QSGD_{10}$", markevery=2)
plt.plot(x, NIID_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, NIID_work, marker='^', label="OurWork", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, nloc=2)
plt.title('(b)CIFAR-10_QSGD_NonIID', y=-0.3)

target_acc = min(max(NIID_quan8), max(NIID_quan10), max(NIID_quan12), max(NIID_quan32), max(NIID_work))
traffic_8 = (np.where(np.array(NIID_quan8) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_10 = (np.where(np.array(NIID_quan10) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_12 = (np.where(np.array(NIID_quan12) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_32 = (np.where(np.array(NIID_quan32) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_work = (np.where(np.array(NIID_work) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
print("CIFAR10_NIID, Accu: {}, quan8: {}, quan10: {}, quan12: {}, quan32: {}, work: {}"
      .format(target_acc, traffic_8, traffic_10, traffic_12, traffic_32, traffic_work))


IID_quan12 = []
with open('../CIFAR100_IID_QSGD/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan12.append(float(line[line.find('y') + 3:]))
# print(IID_quan12)

IID_quan8 = []
with open('../CIFAR100_IID_QSGD/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan8.append(float(line[line.find('y') + 3:]))
# print(IID_quan8)

IID_quan10 = []
with open('../CIFAR100_IID_QSGD/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan10.append(float(line[line.find('y') + 3:]))
# print(IID_quan10)

IID_quan32 = []
with open('../CIFAR100_IID_QSGD/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan32.append(float(line[line.find('y') + 3:]))
# print(IID_quan32)

IID_work = []
with open('../CIFAR100_IID_QSGD/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_work.append(float(line[line.find('y') + 3:]))
# print(IID_work)

start_index = 0
end_index = 14
x = [i for i in range(10, 210, 10)]
x = x[start_index:end_index]
IID_quan12 = IID_quan12[start_index:end_index]
IID_quan8 = IID_quan8[start_index:end_index]
IID_quan10 = IID_quan10[start_index:end_index]
IID_quan32 = IID_quan32[start_index:end_index]
IID_work = IID_work[start_index:end_index]
base = max(max(IID_quan12), max(IID_quan8), max(IID_quan10), max(IID_quan32))
print("Accu improve ", (max(IID_work) - base) / base)
plt.subplot(233)
plt.xlabel("全局迭代", size=13, fontproperties="SimSun")
plt.ylabel("模型性能(%)", size=13, fontproperties="SimSun")
plt.plot(x, IID_quan8, marker='D', label="$P_8Top_k$", markevery=2)
plt.plot(x, IID_quan10, marker='+', label="$P_{10}Top_k$", markevery=2)
plt.plot(x, IID_quan12, marker='*', label="$P_{12}Top_k$", markevery=2)
plt.plot(x, IID_quan32, marker='o', label="$Top_k$", markevery=2)
plt.plot(x, IID_work, marker='^', label="Fed-CVLC", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, ncol=3)
plt.title('(c)CIFAR-100_QSGD_IID', y=-0.3)

target_acc = min(max(IID_quan8), max(IID_quan10), max(IID_quan12), max(IID_quan32), max(IID_work))
traffic_8 = (np.where(np.array(IID_quan8) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_10 = (np.where(np.array(IID_quan10) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_12 = (np.where(np.array(IID_quan12) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_32 = (np.where(np.array(IID_quan32) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_work = (np.where(np.array(IID_work) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
print("CIFAR100_IID, Accu: {}, quan8: {}, quan10: {}, quan12: {}, quan32: {}, work: {}"
      .format(target_acc, traffic_8, traffic_10, traffic_12, traffic_32, traffic_work))



NIID_quan12 = []
with open('../CIFAR100_NIID_QSGD/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan12.append(float(line[line.find('y') + 3:]))
# print(NIID_quan12)

NIID_quan8 = []
with open('../CIFAR100_NIID_QSGD/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan8.append(float(line[line.find('y') + 3:]))
# print(NIID_quan8)

NIID_quan10 = []
with open('../CIFAR100_NIID_QSGD/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan10.append(float(line[line.find('y') + 3:]))
# print(NIID_quan10)

NIID_quan32 = []
with open('../CIFAR100_NIID_QSGD/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan32.append(float(line[line.find('y') + 3:]))
# print(NIID_quan32)

NIID_work = []
with open('../CIFAR100_NIID_QSGD/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_work.append(float(line[line.find('y') + 3:]))
# print(NIID_work)

start_index = 0
end_index = 14
x = [i for i in range(10, 210, 10)]
x = x[start_index:end_index]
NIID_quan12 = NIID_quan12[start_index:end_index]
NIID_quan8 = NIID_quan8[start_index:end_index]
NIID_quan10 = NIID_quan10[start_index:end_index]
NIID_quan32 = NIID_quan32[start_index:end_index]
NIID_work = NIID_work[start_index:end_index]
base = max(max(NIID_quan12), max(NIID_quan8), max(NIID_quan10), max(NIID_quan32))
print("Accu improve ", (max(NIID_work) - base) / base)
# fig = plt.figure(figsize=(5, 3))
plt.subplot(234)
plt.xlabel("全局迭代", size=13, fontproperties="SimSun")
plt.ylabel("模型性能(%)", size=13, fontproperties="SimSun")
plt.plot(x, NIID_quan12, marker='*', label="Spar. with $QSGD_{12}$", markevery=2)
plt.plot(x, NIID_quan8, marker='D', label="Spar. with $QSGD_8$", markevery=2)
plt.plot(x, NIID_quan10, marker='+', label="Spar. with $QSGD_{10}$", markevery=2)
plt.plot(x, NIID_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, NIID_work, marker='^', label="OurWork", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, nloc=2)
plt.title('(d)CIFAR-100_QSGD_NonIID', y=-0.3)

target_acc = min(max(NIID_quan8), max(NIID_quan10), max(NIID_quan12), max(NIID_quan32), max(NIID_work))
traffic_8 = (np.where(np.array(NIID_quan8) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_10 = (np.where(np.array(NIID_quan10) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_12 = (np.where(np.array(NIID_quan12) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_32 = (np.where(np.array(NIID_quan32) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
traffic_work = (np.where(np.array(NIID_work) >= target_acc)[0][0] + 1) * 10 * 90 * 1200 * 10 / 1024 / 1024
print("CIFAR100_NIID, Accu: {}, quan8: {}, quan10: {}, quan12: {}, quan32: {}, work: {}"
      .format(target_acc, traffic_8, traffic_10, traffic_12, traffic_32, traffic_work))


x = [i for i in range(5, 105, 5)]

QSGD_quan12 = []
with open('../FEMNIST_QSGD_result/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan12.append(float(line[line.find('y') + 3:]))
# print(QSGD_quan12)

QSGD_quan8 = []
with open('../FEMNIST_QSGD_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan8.append(float(line[line.find('y') + 3:]))
# print(QSGD_quan8)

QSGD_quan10 = []
with open('../FEMNIST_QSGD_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan10.append(float(line[line.find('y') + 3:]))
# print(QSGD_quan10)

QSGD_quan32 = []
with open('../FEMNIST_QSGD_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan32.append(float(line[line.find('y') + 3:]))
# print(QSGD_quan32)

QSGD_work = []
with open('../FEMNIST_QSGD_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_work.append(float(line[line.find('y') + 3:]))
# print(QSGD_work)
start_index = 0
end_index = 16
x = [i for i in range(5, 105, 5)]
x = x[start_index:end_index]
QSGD_quan12 = QSGD_quan12[start_index:end_index]
QSGD_quan8 = QSGD_quan8[start_index:end_index]
QSGD_quan10 = QSGD_quan10[start_index:end_index]
QSGD_quan32 = QSGD_quan32[start_index:end_index]
QSGD_work = QSGD_work[start_index:end_index]
base = max(max(QSGD_quan12), max(QSGD_quan8), max(QSGD_quan10), max(QSGD_quan32))
print("Accu improve ", (max(QSGD_work) - base) / base)
plt.subplot(235)
plt.xlabel("全局迭代", size=13, fontproperties="SimSun")
plt.ylabel("模型性能(%)", size=13, fontproperties="SimSun")
plt.plot(x, QSGD_quan12, marker='*', label="$P_{12}Top_k$", markevery=2)
plt.plot(x, QSGD_quan8, marker='D', label="$P_8Top_k$", markevery=2)
plt.plot(x, QSGD_quan10, marker='+', label="$P_{10}Top_k$", markevery=2)
plt.plot(x, QSGD_quan32, marker='o', label="$Top_k$", markevery=2)
plt.plot(x, QSGD_work, marker='^', label="Fed-CVLC", markevery=2)
plt.ylim((50))
plt.title('(e)FEMNIST_QSGD_NonIID', y=-0.3)
# plt.legend(loc='best')

target_acc = min(max(QSGD_quan8), max(QSGD_quan10), max(QSGD_quan12), max(QSGD_quan32), max(QSGD_work))
traffic_8 = (np.where(np.array(QSGD_quan8) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_10 = (np.where(np.array(QSGD_quan10) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_12 = (np.where(np.array(QSGD_quan12) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_32 = (np.where(np.array(QSGD_quan32) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
traffic_work = (np.where(np.array(QSGD_work) >= target_acc)[0][0] + 1) * 5 * 10 * 1200 * 10 / 1024 / 1024
print("FEMNIST, Accu: {}, quan8: {}, quan10: {}, quan12: {}, quan32: {}, work: {}"
      .format(target_acc, traffic_8, traffic_10, traffic_12, traffic_32, traffic_work))

plt.show()