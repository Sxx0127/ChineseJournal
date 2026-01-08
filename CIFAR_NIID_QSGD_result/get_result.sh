rm -rf work.txt
cat ../Work/CIFAR/NIID_QSGD.out | grep "Accu" > work.txt
cat ../quan/CIFAR/NIID_QSGD_8.out | grep "Accu" > quan8.txt
cat ../quan/CIFAR/NIID_QSGD_10.out | grep "Accu" > quan10.txt
cat ../quan/CIFAR/NIID_QSGD_12.out | grep "Accu" > quan12.txt
cat ../quan/CIFAR/NIID_QSGD_32.out | grep "Accu" > quan32.txt