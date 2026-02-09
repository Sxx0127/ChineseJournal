rm -rf *txt
cat topk_AB.out | grep "eval_acc" > topk_AB.txt
cat l2STopK.out | grep "eval_acc" > l2STopK.txt
cat updateW.out | grep "eval_acc" > updateW.txt
cat optim.out | grep "eval_acc" > optim.txt

cat l2STopK_7B.out | grep "eval_acc" > l2STopK_7B.txt
cat updateW_7B.out | grep "eval_acc" > updateW_7B.txt


cat topk_AB_03B.out | grep "eval_acc" > topk_AB_03B.txt
cat l2STopK_03B.out | grep "eval_acc" > l2STopK_03B.txt
cat optim_03B.out | grep "eval_acc" > optim_03B.txt


cat topk_AB_05B.out | grep "eval_acc" > topk_AB.txt
cat new_05B.out | grep "eval_acc" > new.txt
cat STopK_05B.out | grep "eval_acc" > STopK.txt
cat 2STopK_05B.out | grep "eval_acc" > 2STopK.txt
cat raw.out | grep "eval_acc" > raw.txt

cat topk_AB.out | grep "Train time" > topk_AB_time.txt
cat new.out | grep "Train time" > new_time.txt
cat topk_AB_05B.out | grep "Train time" > topk_AB_time.txt
cat new_05B.out | grep "Train time" > new_time.txt
cat topk_AB_7B.out | grep "Train time" > topk_AB_time.txt
cat new_7B.out | grep "Train time" > new_time.txt
