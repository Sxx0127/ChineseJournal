import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.use('TkAgg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
        }


# fig = plt.figure(figsize=(5.7, 3.7))
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
models = ['roberta_20news_iid', 'roberta_20news_niid', '3llama_20news_iid', '3llama_20news_niid', 
         'llama_20news_iid', 'llama_20news_niid']
titles = ['RoBERTa(0.3B) IID', 'RoBERTa(0.3B) non-IID', 'LLaMA 3.2(1B) IID', 'LLaMA 3.2(1B) non-IID',
          'LLaMA 2(7B) IID', 'LLaMA 2(7B) non-IID']
traffic = [140 * 1500 / 1024 / 1024, 140 * 1500 / 1024 / 1024, 152 * 1500 / 1024 / 1024, 152 * 1500 / 1024 / 1024,
           782 * 1500 / 1024 / 1024, 782 * 1500 / 1024 / 1024]

# model = 'distilbert_20news_iid'
# model = 'largeroberta_20news_iid'
# model = '3llama_20news_iid'

raw = []
with open('accu/distilbert_20news_iid_raw1.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        raw.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
new5 = []
with open('accu/distilbert_20news_iid_new5.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        new5.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
new6 = []
with open('accu/distilbert_20news_iid_new6.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        new6.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
new7 = []
with open('accu/distilbert_20news_iid_new7.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        new7.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
new8 = []
with open('accu/distilbert_20news_iid_new8.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        new8.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
new9 = []
with open('accu/distilbert_20news_iid_new9.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        new9.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))

inter = 1
x = range(1, 101)
raw = raw[:len(x)]
new5 = new5[:len(x)]
new6 = new6[:len(x)]
new7 = new7[:len(x)]
new8 = new8[:len(x)]
new9 = new9[:len(x)]

axs.set_xlabel("Global Iterations", size=15, family='Times New Roman')
axs.set_ylabel("Accuracy", size=15, family='Times New Roman')
markevery = 5
axs.plot(x[:len(raw)], raw, marker='x', markersize=5, label="raw", markevery=markevery, markerfacecolor='white')
axs.plot(x[:len(new5)], new5, marker='*', markersize=5, label="new5", markevery=markevery, markerfacecolor='white')
axs.plot(x[:len(new6)], new6, marker='D', markersize=5, label="new6", markevery=markevery, markerfacecolor='white')
axs.plot(x[:len(new7)], new7, marker='^', markersize=5, label="new7", markevery=markevery, markerfacecolor='white')
axs.plot(x[:len(new8)], new8, marker='+', markersize=5, label="new8", markevery=markevery, markerfacecolor='white')
axs.plot(x[:len(new9)], new9, marker='.', markersize=5, label="new9", markevery=markevery, markerfacecolor='white')
# plt.title('roberta-large (0.3B)')
axs.set_ylim(0.3,)
axs.legend(bbox_to_anchor=(0, 1.07), loc=6, ncol=6, borderaxespad=0, prop=font1)

# target_accu = int(max(compeft) * 100)
# updateW_tra = 10 * (np.where(np.array(updateW) >= target_accu / 100) + 1) * traffic[i]
# optim_tra = 10 * (np.where(np.array(optim) >= target_accu / 100) + 1) * traffic[i]
# block_tra = 10 * (np.where(np.array(block) >= target_accu / 100) + 1) * traffic[i]
# topk_tra = 10 * (np.where(np.array(topk) >= target_accu / 100) + 1) * traffic[i]
# compeft_tra = 10 * (np.where(np.array(compeft) >= target_accu / 100) + 1) * traffic[i]
# print("Target: {}, Iteration: {}, optim: {}, UpdateW: {}, block: {}, topk: {}, Compeft: {}, Reduce: {}".
#       format(target_accu, i, optim_tra, updateW_tra, block_tra, topk_tra, compeft_tra, 
#              (block_tra-optim_tra) / block_tra), )
# print()
plt.show()
