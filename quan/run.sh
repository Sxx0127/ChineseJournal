rm -rf CIFAR/*PQ*
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 6 > CIFAR/IID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 8 > CIFAR/IID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 6 > CIFAR/NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 8 > CIFAR/NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 10 > CIFAR/IID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 32 > CIFAR/IID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 10 > CIFAR/NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 32 > CIFAR/NIID_PQ_32.out 2>&1 &

rm -rf CIFAR/*QSGD*
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 8 > CIFAR/IID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 10 > CIFAR/IID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 8 > CIFAR/NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 10 > CIFAR/NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 12 > CIFAR/IID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 32 > CIFAR/IID_QSGD_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 12 > CIFAR/NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 32 > CIFAR/NIID_QSGD_32.out 2>&1 &


rm -rf CIFAR100/*PQ*.out
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 6 > CIFAR100/IID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 8 > CIFAR100/IID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 6 > CIFAR100/NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 8 > CIFAR100/NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 10 > CIFAR100/IID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 32 > CIFAR100/IID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 10 > CIFAR100/NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 32 > CIFAR100/NIID_PQ_32.out 2>&1 &

rm -rf CIFAR100/*QSGD*.out
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 8 > CIFAR100/IID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 10 > CIFAR100/IID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 8 > CIFAR100/NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 10 > CIFAR100/NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 12 > CIFAR100/IID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 32 > CIFAR100/IID_QSGD_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 12 > CIFAR100/NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 32 > CIFAR100/NIID_QSGD_32.out 2>&1 &

rm -rf FEMNIST/*
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 6 > FEMNIST/NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 8 > FEMNIST/NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 10 > FEMNIST/NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 32 > FEMNIST/NIID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 8 > FEMNIST/NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 10 > FEMNIST/NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 12 > FEMNIST/NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 32 > FEMNIST/NIID_QSGD_32.out 2>&1 &

nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 6 > shakespeare_NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 8 > shakespeare_NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 10 > shakespeare_NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 32 > shakespeare_NIID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 6 > shakespeare_NIID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 8 > shakespeare_NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 10 > shakespeare_NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 12 > shakespeare_NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 32 > shakespeare_NIID_QSGD_32.out 2>&1 &






nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 6 > fedprox/CIFAR_IID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 8 > fedprox/CIFAR_IID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 10 > fedprox/CIFAR_IID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ --bit 32 > fedprox/CIFAR_IID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 6 > fedprox/CIFAR_NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 8 > fedprox/CIFAR_NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 10 > fedprox/CIFAR_NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ --bit 32 > fedprox/CIFAR_NIID_PQ_32.out 2>&1 &

nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 6 > fedprox/CIFAR_IID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 8 > fedprox/CIFAR_IID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 10 > fedprox/CIFAR_IID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 12 > fedprox/CIFAR_IID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD --bit 32 > fedprox/CIFAR_IID_QSGD_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 6 > fedprox/CIFAR_NIID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 8 > fedprox/CIFAR_NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 10 > fedprox/CIFAR_NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 12 > fedprox/CIFAR_NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD --bit 32 > fedprox/CIFAR_NIID_QSGD_32.out 2>&1 &

nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 6 > fedprox/CIFAR100_IID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 8 > fedprox/CIFAR100_IID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 10 > fedprox/CIFAR100_IID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ --bit 32 > fedprox/CIFAR100_IID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 6 > fedprox/CIFAR100_NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 8 > fedprox/CIFAR100_NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 10 > fedprox/CIFAR100_NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ --bit 32 > fedprox/CIFAR100_NIID_PQ_32.out 2>&1 &

nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 6 > fedprox/CIFAR100_IID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 8 > fedprox/CIFAR100_IID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 10 > fedprox/CIFAR100_IID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 12 > fedprox/CIFAR100_IID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD --bit 32 > fedprox/CIFAR100_IID_QSGD_32.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 6 > fedprox/CIFAR100_NIID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 8 > fedprox/CIFAR100_NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 10 > fedprox/CIFAR100_NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 12 > fedprox/CIFAR100_NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD --bit 32 > fedprox/CIFAR100_NIID_QSGD_32.out 2>&1 &

nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 6 > fedprox/FEMNIST_NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 8 > fedprox/FEMNIST_NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 10 > fedprox/FEMNIST_NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ --bit 32 > fedprox/FEMNIST_NIID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 6 > fedprox/FEMNIST_NIID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 8 > fedprox/FEMNIST_NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 10 > fedprox/FEMNIST_NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 12 > fedprox/FEMNIST_NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD --bit 32 > fedprox/FEMNIST_NIID_QSGD_32.out 2>&1 &

nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 6 > shakespeare_NIID_PQ_6.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 8 > shakespeare_NIID_PQ_8.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 10 > shakespeare_NIID_PQ_10.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan PQ --bit 32 > shakespeare_NIID_PQ_32.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 6 > shakespeare_NIID_QSGD_6.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 8 > shakespeare_NIID_QSGD_8.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 10 > shakespeare_NIID_QSGD_10.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 12 > shakespeare_NIID_QSGD_12.out 2>&1 &
nohup python -u Server.py --dataset shakespeare --iid NIID --quan QSGD --bit 32 > shakespeare_NIID_QSGD_32.out 2>&1 &
