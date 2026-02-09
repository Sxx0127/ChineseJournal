import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }

fig = plt.figure(figsize=(6, 3))

# topk = []
# with open('topk_AB.txt', encoding='utf-8') as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         topk.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]) * 100)

# new = []
# with open('new.txt', encoding='utf-8') as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         new.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]) * 100)

l2STopK = []
with open('l2STopK_7B.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        l2STopK.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]) * 100)
updateW = []
with open('updateW_7B.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        updateW.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]) * 100)

# optim = []
# with open('optim.txt', encoding='utf-8') as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         optim.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]) * 100)

# inter = 1
# x = list(map(str, range(1, 21)))
# raw = raw[:len(x)]
# topk = topk[:len(x)]
# randk = randk[:len(x)]
# new = new[:len(x)]
# plt.xlabel("Global Round", size=15, family='Times New Roman')
# plt.ylabel("Accuracy (%)", size=15, family='Times New Roman')
# plt.plot(x, raw, marker='*', markersize=8, label="Uncompressed", markevery=1, markerfacecolor='white')
# plt.plot(x, topk, marker='D', markersize=8, label="topk", markevery=1, markerfacecolor='white')
# plt.plot(x, randk, marker='^', markersize=8, label="randk", markevery=1, markerfacecolor='white')
# # plt.plot(x, new, marker='x', markersize=7, label="new", markevery=1, markerfacecolor='white',
# #             markeredgewidth=2)
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, ncol=5, borderaxespad=0, prop=font1)


inter = 1
x = list(range(1, 6))
length = len(x)
# topk = topk[:len(x)]
x = x[0:length]
l2STopK = l2STopK[0:length]
updateW = updateW[0:length]
# optim = optim[:len(x)]
plt.xlabel("Global Round", size=15, family='Times New Roman')
plt.ylabel("Accuracy (%)", size=15, family='Times New Roman')
# plt.plot(x, raw, marker='*', markersize=8, label="A,B", markevery=1, markerfacecolor='white')
# plt.plot(x, topk, marker='D', markersize=8, label="topk(A),topk(B)", markevery=1, markerfacecolor='white')
plt.plot(x, l2STopK, marker='^', markersize=8, label="old", markevery=1, markerfacecolor='white')
plt.plot(x, updateW, marker='^', markersize=8, label="new", markevery=1, markerfacecolor='white')
# plt.plot(x, optim, marker='^', markersize=8, label="topk(optimize)", markevery=1, markerfacecolor='white')
# plt.legend(bbox_to_anchor=(0, 1.05), loc=3, ncol=5, borderaxespad=0, prop=font1)
plt.legend(loc='best')
# plt.title("distilbert-base-multilingual-cased (0.1B)")
# plt.title("roberta-large (0.3B)")
plt.title("llama-2-7B (7B)")

plt.show()
