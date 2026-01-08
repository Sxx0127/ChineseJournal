import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

top1_8 = []
with open('top1_8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top1_8.append(float(line[line.find('y') + 3:]))
print(top1_8)

top2_4 = []
with open('top2_4.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top2_4.append(float(line[line.find('y') + 3:]))
print(top2_4)

top4_2 = []
with open('top4_2.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top4_2.append(float(line[line.find('y') + 3:]))
print(top4_2)

top15_5 = []
with open('top15_5.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top15_5.append(float(line[line.find('y') + 3:]))
print(top15_5)

top25_3 = []
with open('top25_3.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top25_3.append(float(line[line.find('y') + 3:]))
print(top25_3)

start_index = 0
end_index = 23
x = [i for i in range(start_index + 1, end_index + 1, 1)]
x = x[start_index:end_index]
top1_8 = top1_8[start_index:end_index]
top2_4 = top2_4[start_index:end_index]
top4_2 = top4_2[start_index:end_index]
top15_5 = top15_5[start_index:end_index]
top25_3 = top25_3[start_index:end_index]
plt.plot(x, top1_8, marker='*', label="Top 1%, 8 bits", markevery=1)
plt.plot(x, top15_5, marker='+', label="Top 1.5%, 5 bits", markevery=1)
plt.plot(x, top2_4, marker='D', label="Top 2%, 4 bits", markevery=1)
# plt.plot(x, top4_2, marker='^', label="Top 4%, 2 bits", markevery=1)
plt.plot(x, top25_3, marker='', label="Top 2.5%, 3 bits", markevery=1)
plt.xlabel('#Rounds')
plt.ylabel('Test Accuracy')
# plt.ylim(10)
# plt.title('ResNet-18')
plt.legend(loc='best')
plt.show()
