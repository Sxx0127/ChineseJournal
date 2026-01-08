import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

x = [i for i in range(5, 105, 5)]

IID_quan6 = []
with open('../CIFAR_IID_PQ_result/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan6.append(float(line[line.find('y') + 3:]))
print(IID_quan6)

IID_quan8 = []
with open('../CIFAR_IID_PQ_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan8.append(float(line[line.find('y') + 3:]))
print(IID_quan8)

IID_quan10 = []
with open('../CIFAR_IID_PQ_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan10.append(float(line[line.find('y') + 3:]))
print(IID_quan10)

IID_quan32 = []
with open('../CIFAR_IID_PQ_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan32.append(float(line[line.find('y') + 3:]))
print(IID_quan32)

IID_work = []
with open('../CIFAR_IID_PQ_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_work.append(float(line[line.find('y') + 3:]))
print(IID_work)
end_index = 16
x = x[:end_index]
IID_quan6 = IID_quan6[:end_index]
IID_quan8 = IID_quan8[:end_index]
IID_quan10 = IID_quan10[:end_index]
IID_quan32 = IID_quan32[:end_index]
IID_work = IID_work[:end_index]
base = max(max(IID_quan6), max(IID_quan8), max(IID_quan10), max(IID_quan32))
print("Accu improve ", (max(IID_work) - base) / base)
fig = plt.figure(figsize=(12, 4))
plt.subplot(141)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, IID_quan6, marker='*', label="$P_6Top_k$", markevery=2)
plt.plot(x, IID_quan8, marker='D', label="$P_8Top_k$", markevery=2)
plt.plot(x, IID_quan10, marker='+', label="$P_{10}Top_k$", markevery=2)
plt.plot(x, IID_quan32, marker='o', label="$Top_k$", markevery=2)
plt.plot(x, IID_work, marker='^', label="Fed-CVLC", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, ncol=3)
plt.title('(a)CIFAR-10_PQ_IID', y=-0.3)

NIID_quan6 = []
with open('../CIFAR_NIID_PQ_result/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan6.append(float(line[line.find('y') + 3:]))
print(NIID_quan6)

NIID_quan8 = []
with open('../CIFAR_NIID_PQ_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan8.append(float(line[line.find('y') + 3:]))
print(NIID_quan8)

NIID_quan10 = []
with open('../CIFAR_NIID_PQ_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan10.append(float(line[line.find('y') + 3:]))
print(NIID_quan10)

NIID_quan32 = []
with open('../CIFAR_NIID_PQ_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan32.append(float(line[line.find('y') + 3:]))
print(NIID_quan32)

NIID_work = []
with open('../CIFAR_NIID_PQ_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_work.append(float(line[line.find('y') + 3:]))
print(NIID_work)

NIID_quan6 = NIID_quan6[:end_index]
NIID_quan8 = NIID_quan8[:end_index]
NIID_quan10 = NIID_quan10[:end_index]
NIID_quan32 = NIID_quan32[:end_index]
NIID_work = NIID_work[:end_index]
base = max(max(NIID_quan6), max(NIID_quan8), max(NIID_quan10), max(NIID_quan32))
print("Accu improve ", (max(NIID_work) - base) / base)
# fig = plt.figure(figsize=(5, 3))
plt.subplot(142)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, NIID_quan6, marker='*', label="Spar. with $PQ_6$", markevery=2)
plt.plot(x, NIID_quan8, marker='D', label="Spar. with $PQ_8$", markevery=2)
plt.plot(x, NIID_quan10, marker='+', label="Spar. with $PQ_{10}$", markevery=2)
plt.plot(x, NIID_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, NIID_work, marker='^', label="OurWork", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, nloc=2)
plt.title('(b)CIFAR-10_PQ_non-IID', y=-0.3)

IID_quan6 = []
with open('../CIFAR_IID_QSGD_result/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan6.append(float(line[line.find('y') + 3:]))
print(IID_quan6)

IID_quan8 = []
with open('../CIFAR_IID_QSGD_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan8.append(float(line[line.find('y') + 3:]))
print(IID_quan8)

IID_quan10 = []
with open('../CIFAR_IID_QSGD_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan10.append(float(line[line.find('y') + 3:]))
print(IID_quan10)

IID_quan12 = []
with open('../CIFAR_IID_QSGD_result/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan12.append(float(line[line.find('y') + 3:]))
print(IID_quan12)

IID_quan32 = []
with open('../CIFAR_IID_QSGD_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan32.append(float(line[line.find('y') + 3:]))
print(IID_quan32)

IID_work = []
with open('../CIFAR_IID_QSGD_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_work.append(float(line[line.find('y') + 3:]))
print(IID_work)

end_index = 16
x = x[:end_index]
IID_quan6 = IID_quan6[:end_index]
IID_quan8 = IID_quan8[:end_index]
IID_quan10 = IID_quan10[:end_index]
IID_quan12 = IID_quan12[:end_index]
IID_quan32 = IID_quan32[:end_index]
IID_work = IID_work[:end_index]
base = max(max(IID_quan6), max(IID_quan8), max(IID_quan10), max(IID_quan32))
print("Accu improve ", (max(IID_work) - base) / base)
plt.subplot(143)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, IID_quan6, marker='*', label="$Q_6Top_k$", markevery=2)
plt.plot(x, IID_quan8, marker='D', label="$Q_8Top_k$", markevery=2)
plt.plot(x, IID_quan10, marker='+', label="$Q_{10}Top_k$", markevery=2)
# plt.plot(x, IID_quan12, marker='*', label="$Q_{12}Top_k$", markevery=2)
plt.plot(x, IID_quan32, marker='o', label="$Top_k$", markevery=2)
plt.plot(x, IID_work, marker='^', label="Fed-CVLC", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, ncol=3)
plt.title('(c)CIFAR-10_QSGD_IID', y=-0.3)

NIID_quan6 = []
with open('../CIFAR_NIID_QSGD_result/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan6.append(float(line[line.find('y') + 3:]))
print(NIID_quan6)

NIID_quan8 = []
with open('../CIFAR_NIID_QSGD_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan8.append(float(line[line.find('y') + 3:]))
print(NIID_quan8)

NIID_quan10 = []
with open('../CIFAR_NIID_QSGD_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan10.append(float(line[line.find('y') + 3:]))
print(NIID_quan10)

NIID_quan12 = []
with open('../CIFAR_NIID_QSGD_result/quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan12.append(float(line[line.find('y') + 3:]))
print(NIID_quan12)

NIID_quan32 = []
with open('../CIFAR_NIID_QSGD_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan32.append(float(line[line.find('y') + 3:]))
print(NIID_quan32)

NIID_work = []
with open('../CIFAR_NIID_QSGD_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_work.append(float(line[line.find('y') + 3:]))
print(NIID_work)
NIID_quan6 = NIID_quan6[:end_index]
NIID_quan8 = NIID_quan8[:end_index]
NIID_quan10 = NIID_quan10[:end_index]
NIID_quan12 = NIID_quan12[:end_index]
NIID_quan32 = NIID_quan32[:end_index]
NIID_work = NIID_work[:end_index]
base = max(max(NIID_quan6), max(NIID_quan8), max(NIID_quan10), max(NIID_quan32))
print("Accu improve ", (max(NIID_work) - base) / base)
# fig = plt.figure(figsize=(5, 3))
plt.subplot(144)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, NIID_quan6, marker='*', label="Spar. with $QSGD_6$", markevery=2)
plt.plot(x, NIID_quan8, marker='D', label="Spar. with $QSGD_8$", markevery=2)
plt.plot(x, NIID_quan10, marker='+', label="Spar. with $QSGD_{10}$", markevery=2)
# plt.plot(x, NIID_quan12, marker='*', label="Spar. with $QSGD_{12}$", markevery=2)
plt.plot(x, NIID_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, NIID_work, marker='^', label="OurWork", markevery=2)
plt.ylim((30))
plt.title('(d)CIFAR-10_QSGD_non-IID', y=-0.3)
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, nloc=2)


plt.show()
