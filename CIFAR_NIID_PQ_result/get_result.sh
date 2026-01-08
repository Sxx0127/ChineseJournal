rm -rf *.txt
cat ../Work/CIFAR/NIID_PQ.out | grep "Accu" > work.txt
# cat ../Work/nohup.out | grep "Accu" > work.txt
cat ../quan/CIFAR/NIID_PQ_6.out | grep "Accu" > quan6.txt
cat ../quan/CIFAR/NIID_PQ_8.out | grep "Accu" > quan8.txt
cat ../quan/CIFAR/NIID_PQ_10.out | grep "Accu" > quan10.txt
cat ../quan/CIFAR/NIID_PQ_32.out | grep "Accu" > quan32.txt
