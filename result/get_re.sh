cat 3llama_20news_iid_optim.out | grep "eval_acc" > accu/3llama_20news_iid_optim.txt
cat 3llama_20news_iid_compeft.out | grep "eval_acc" > accu/3llama_20news_iid_compeft.txt
cat 3llama_20news_iid_prune.out | grep "eval_acc" > accu/3llama_20news_iid_prune.txt
cat 3llama_20news_iid_topk.out | grep "eval_acc" > accu/3llama_20news_iid_topk_AB.txt
cat 3llama_20news_iid_block.out | grep "eval_acc" > accu/3llama_20news_iid_block.txt
cat 3llama_20news_iid_updateW.out | grep "eval_acc" > accu/3llama_20news_iid_updateW.txt

cat 3llama_20news_niid_optim.out | grep "eval_acc" > accu/3llama_20news_niid_optim.txt
cat 3llama_20news_niid_compeft.out | grep "eval_acc" > accu/3llama_20news_niid_compeft.txt
cat 3llama_20news_niid_prune.out | grep "eval_acc" > accu/3llama_20news_niid_prune.txt
cat 3llama_20news_niid_topk.out | grep "eval_acc" > accu/3llama_20news_niid_topk_AB.txt
cat 3llama_20news_niid_block.out | grep "eval_acc" > accu/3llama_20news_niid_block.txt
cat 3llama_20news_niid_updateW.out | grep "eval_acc" > accu/3llama_20news_niid_updateW.txt


cat llama_20news_iid_optim.out | grep "eval_acc" > accu/llama_20news_iid_optim.txt
cat llama_20news_iid_compeft.out | grep "eval_acc" > accu/llama_20news_iid_compeft.txt
cat llama_20news_iid_prune.out | grep "eval_acc" > accu/llama_20news_iid_prune.txt
cat llama_20news_iid_topk.out | grep "eval_acc" > accu/llama_20news_iid_topk_AB.txt
cat llama_20news_iid_block.out | grep "eval_acc" > accu/llama_20news_iid_block.txt
cat llama_20news_iid_updateW.out | grep "eval_acc" > accu/llama_20news_iid_updateW.txt

cat llama_20news_niid_optim.out | grep "eval_acc" > accu/llama_20news_niid_optim.txt
cat llama_20news_niid_compeft.out | grep "eval_acc" > accu/llama_20news_niid_compeft.txt
cat llama_20news_niid_prune.out | grep "eval_acc" > accu/llama_20news_niid_prune.txt
cat llama_20news_niid_topk.out | grep "eval_acc" > accu/llama_20news_niid_topk_AB.txt
cat llama_20news_niid_updateW.out | grep "eval_acc" > accu/llama_20news_niid_updateW.txt
cat llama_20news_niid_block.out | grep "eval_acc" > accu/llama_20news_niid_block.txt

cat roberta_20news_iid_optim.out | grep "eval_acc" > accu/roberta_20news_iid_optim.txt
cat roberta_20news_iid_topk.out | grep "eval_acc" > accu/roberta_20news_iid_topk_AB.txt
cat roberta_20news_iid_prune.out | grep "eval_acc" > accu/roberta_20news_iid_prune.txt
cat roberta_20news_iid_updateW.out | grep "eval_acc" > accu/roberta_20news_iid_updateW.txt
cat roberta_20news_iid_compeft.out | grep "eval_acc" > accu/roberta_20news_iid_compeft.txt
cat roberta_20news_iid_block.out | grep "eval_acc" > accu/roberta_20news_iid_block.txt

cat roberta_20news_niid_optim.out | grep "eval_acc" > accu/roberta_20news_niid_optim.txt
cat roberta_20news_niid_topk.out | grep "eval_acc" > accu/roberta_20news_niid_topk_AB.txt
cat roberta_20news_niid_prune.out | grep "eval_acc" > accu/roberta_20news_niid_prune.txt
cat roberta_20news_niid_updateW.out | grep "eval_acc" > accu/roberta_20news_niid_updateW.txt
cat roberta_20news_niid_compeft.out | grep "eval_acc" > accu/roberta_20news_niid_compeft.txt
cat roberta_20news_niid_block.out | grep "eval_acc" > accu/roberta_20news_niid_block.txt

