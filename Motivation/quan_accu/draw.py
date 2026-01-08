import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

uniform = []
with open('uniform.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        uniform.append(float(line[line.find('y') + 3:]))
print(uniform)

non_uniform_262 = []
with open('non-uniform_262.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        non_uniform_262.append(float(line[line.find('y') + 3:]))
print(non_uniform_262)

non_uniform_343 = []
with open('non-uniform_343.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        non_uniform_343.append(float(line[line.find('y') + 3:]))
print(non_uniform_343)


start_index = 30
end_index = 55
x = [i for i in range(start_index+1, end_index+1, 1)]
# x = x[start_index:end_index]
uniform = uniform[start_index:end_index]
non_uniform_262 = non_uniform_262[start_index:end_index]
non_uniform_343 = non_uniform_343[start_index:end_index]
plt.plot(x, uniform, marker='*', label="Uniform Quantization", markevery=1)
plt.plot(x, non_uniform_262, marker='D', label="Non-Uniform Quantization (262)", markevery=2)
plt.plot(x, non_uniform_343, marker='+', label="Non-Uniform Quantization (343)", markevery=3)
plt.xlabel('#Rounds')
plt.ylabel('Test Accuracy')
# plt.ylim(10)
# plt.title('ResNet-18')
plt.legend(loc='best')
plt.show()
