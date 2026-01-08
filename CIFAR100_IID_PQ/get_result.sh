rm -rf *.txt
cat ../Work/CIFAR100/IID_PQ.out | grep "Accu" > work.txt
cat ../quan/CIFAR100/IID_PQ_8.out | grep "Accu" > quan8.txt
cat ../quan/CIFAR100/IID_PQ_6.out | grep "Accu" > quan6.txt
cat ../quan/CIFAR100/IID_PQ_10.out | grep "Accu" > quan10.txt
cat ../quan/CIFAR100/IID_PQ_32.out | grep "Accu" > quan32.txt
