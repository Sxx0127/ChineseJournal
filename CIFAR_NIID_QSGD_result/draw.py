import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# quan6 = []
# with open('quan6.txt', encoding='utf-8') as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         quan6.append(float(line[line.find('y') + 3:]))
# print(quan6)

quan8 = []
with open('quan8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        quan8.append(float(line[line.find('y') + 3:]))
print(quan8)

quan10 = []
with open('quan10.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        quan10.append(float(line[line.find('y') + 3:]))
print(quan10)

quan12 = []
with open('quan12.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        quan12.append(float(line[line.find('y') + 3:]))
print(quan12)

quan32 = []
with open('quan32.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        quan32.append(float(line[line.find('y') + 3:]))
print(quan32)

work = []
with open('work.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        work.append(float(line[line.find('y') + 3:]))
print(work)

start_index = 0
end_index = 20
x = [i for i in range(5, 105, 5)]
x = x[start_index:end_index]
# quan6 = quan6[start_index:end_index]
quan8 = quan8[start_index:end_index]
quan10 = quan10[start_index:end_index]
quan12 = quan12[start_index:end_index]
quan32 = quan32[start_index:end_index]
work = work[start_index:end_index]
# plt.plot(x, quan2, marker='+', label="Spar. with $PQ_2$", markevery=5)
# plt.plot(x, quan6, marker='*', label="Spar. with $PQ_6$", markevery=2)
plt.plot(x[:len(quan8)], quan8, marker='D', label="Spar. with $PQ_8$", markevery=2)
plt.plot(x[:len(quan10)], quan10, marker='+', label="Spar. with $PQ_{10}$", markevery=2)
plt.plot(x[:len(quan12)], quan12, marker='o', label="Spar. with $PQ_{12}$", markevery=2)
plt.plot(x[:len(quan32)], quan32, marker='o', label="Spar.", markevery=2)
plt.plot(x[:len(work)], work, marker='^', label="OurWork", markevery=2)
plt.xlabel('#Rounds')
plt.ylabel('Test Accuracy')
plt.ylim(30)
# plt.title('ResNet-18')
plt.legend(loc='best')
plt.show()
