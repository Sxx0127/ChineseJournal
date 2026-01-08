import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

fit = []
with open('fit.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        fit.append(float(line[line.find('y') + 3:]))
print(fit)

top = []
with open('top.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        top.append(float(line[line.find('y') + 3:]))
print(top)
start_index = 0
end_index = 20
x = [i for i in range(5, 105, 5)]
x = x[start_index:end_index]
fit = fit[start_index:end_index]
top = top[start_index:end_index]
plt.plot(x, fit, marker='*', label="fitted updates", markevery=2)
plt.plot(x, top, marker='D', label="sign updates", markevery=2)
plt.xlabel('#Rounds')
plt.ylabel('Test Accuracy')
# plt.ylim(10)
# plt.title('ResNet-18')
plt.legend(loc='best')
plt.show()
