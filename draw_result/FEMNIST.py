import matplotlib.pyplot as plt

x = [i for i in range(5, 105, 5)]

PQ_quan6 = []
with open('../FEMNIST_PQ_result/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        PQ_quan6.append(float(line[line.find('y') + 3:]))
print(PQ_quan6)

PQ_quan8 = []
with open('../FEMNIST_PQ_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        PQ_quan8.append(float(line[line.find('y') + 3:]))
print(PQ_quan8)

PQ_quan10 = []
with open('../FEMNIST_PQ_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        PQ_quan10.append(float(line[line.find('y') + 3:]))
print(PQ_quan10)

PQ_quan32 = []
with open('../FEMNIST_PQ_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        PQ_quan32.append(float(line[line.find('y') + 3:]))
print(PQ_quan32)

PQ_work = []
with open('../FEMNIST_PQ_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        PQ_work.append(float(line[line.find('y') + 3:]))
print(PQ_work)

fig = plt.figure(figsize=(6, 3.5))
plt.subplot(121)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, PQ_quan6, marker='*', label="Spar. with $PQ_6$", markevery=2)
plt.plot(x, PQ_quan8, marker='D', label="Spar. with $PQ_8$", markevery=2)
plt.plot(x, PQ_quan10, marker='+', label="Spar. with $PQ_{10}$", markevery=2)
plt.plot(x, PQ_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, PQ_work, marker='^', label="OurWork", markevery=2)
plt.ylim((50))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, ncol=3)



QSGD_quan6 = []
with open('../FEMNIST_QSGD_result/quan6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan6.append(float(line[line.find('y') + 3:]))
print(QSGD_quan6)

QSGD_quan8 = []
with open('../FEMNIST_QSGD_result/quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan8.append(float(line[line.find('y') + 3:]))
print(QSGD_quan8)

QSGD_quan10 = []
with open('../FEMNIST_QSGD_result/quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan10.append(float(line[line.find('y') + 3:]))
print(QSGD_quan10)

QSGD_quan32 = []
with open('../FEMNIST_QSGD_result/quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_quan32.append(float(line[line.find('y') + 3:]))
print(QSGD_quan32)

QSGD_work = []
with open('../FEMNIST_QSGD_result/work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        QSGD_work.append(float(line[line.find('y') + 3:]))
print(QSGD_work)

# fig = plt.figure(figsize=(5, 3))
plt.subplot(122)
plt.xlabel("#Global Iterations", size=12)
plt.ylabel("Accuracy(%)", size=12)
plt.plot(x, QSGD_quan6, marker='*', label="Spar. with $PQ_6$", markevery=2)
plt.plot(x, QSGD_quan8, marker='D', label="Spar. with $PQ_8$", markevery=2)
plt.plot(x, QSGD_quan10, marker='+', label="Spar. with $PQ_{10}$", markevery=2)
plt.plot(x, QSGD_quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x, QSGD_work, marker='^', label="OurWork", markevery=2)
plt.ylim((50))
# plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, nloc=2)



plt.show()
