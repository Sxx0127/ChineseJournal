import matplotlib.pyplot as plt

x = [i for i in range(5, 105, 5)]

IID_quan6 = []
with open('../CIFAR100_IID_PQ/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan6.append(float(line[line.find('y') + 3:]))
print(IID_quan6)

IID_quan8 = []
with open('../CIFAR100_IID_PQ/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan8.append(float(line[line.find('y') + 3:]))
print(IID_quan8)

IID_quan10 = []
with open('../CIFAR100_IID_PQ/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan10.append(float(line[line.find('y') + 3:]))
print(IID_quan10)

IID_quan32 = []
with open('../CIFAR100_IID_PQ/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_quan32.append(float(line[line.find('y') + 3:]))
print(IID_quan32)

IID_work = []
with open('../CIFAR100_IID_PQ/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        IID_work.append(float(line[line.find('y') + 3:]))
print(IID_work)

start_index = 0
end_index = 18
x = [i for i in range(5, 105, 5)]
x = x[start_index:end_index]
IID_quan6 = IID_quan6[start_index:end_index]
IID_quan8 = IID_quan8[start_index:end_index]
IID_quan10 = IID_quan10[start_index:end_index]
IID_quan32 = IID_quan32[start_index:end_index]
IID_work = IID_work[start_index:end_index]
fig = plt.figure(figsize=(6, 3.5))
plt.subplot(121)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, IID_quan6, marker='*', label="Spar. with $PQ_6$", markevery=2)
plt.plot(x, IID_quan8, marker='D', label="Spar. with $PQ_8$", markevery=2)
plt.plot(x, IID_quan10, marker='+', label="Spar. with $PQ_{10}$", markevery=2)
plt.plot(x, IID_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, IID_work, marker='^', label="OurWork", markevery=2)
plt.ylim((30))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, ncol=3)



NIID_quan6 = []
with open('../CIFAR100_NIID_PQ/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan6.append(float(line[line.find('y') + 3:]))
print(NIID_quan6)

NIID_quan8 = []
with open('../CIFAR100_NIID_PQ/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan8.append(float(line[line.find('y') + 3:]))
print(NIID_quan8)

NIID_quan10 = []
with open('../CIFAR100_NIID_PQ/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan10.append(float(line[line.find('y') + 3:]))
print(NIID_quan10)

NIID_quan32 = []
with open('../CIFAR100_NIID_PQ/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_quan32.append(float(line[line.find('y') + 3:]))
print(NIID_quan32)

NIID_work = []
with open('../CIFAR100_NIID_PQ/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        NIID_work.append(float(line[line.find('y') + 3:]))
print(NIID_work)

start_index = 0
end_index = 18
x = [i for i in range(5, 105, 5)]
x = x[start_index:end_index]
NIID_quan6 = NIID_quan6[start_index:end_index]
NIID_quan8 = NIID_quan8[start_index:end_index]
NIID_quan10 = NIID_quan10[start_index:end_index]
NIID_quan32 = NIID_quan32[start_index:end_index]
NIID_work = NIID_work[start_index:end_index]
# fig = plt.figure(figsize=(5, 3))
plt.subplot(122)
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



plt.show()
