nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 6 > CIFAR_IID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 8 > CIFAR_IID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 10 > CIFAR_IID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 32 > CIFAR_IID_PQ_32.out 2>&1 &