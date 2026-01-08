lr和batch_size越小越好？

rm -rf CIFAR/*PQ*
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ > CIFAR/IID_PQ.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ > CIFAR/NIID_PQ.out 2>&1 &
rm -rf CIFAR/*QSGD*
nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD > CIFAR/IID_QSGD.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD > CIFAR/NIID_QSGD.out 2>&1 &

rm -rf CIFAR100/*PQ*
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ > CIFAR100/IID_PQ.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ > CIFAR100/NIID_PQ.out 2>&1 &
rm -rf CIFAR100/*QSGD*
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD > CIFAR100/IID_QSGD.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD > CIFAR100/NIID_QSGD.out 2>&1 &

rm -rf FEMNIST/*
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ > FEMNIST/NIID_PQ.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD > FEMNIST/NIID_QSGD.out 2>&1 &



abcd是求绝对值的mean的
直接求mean的
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ > e.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ > f.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD > g.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD > h.out 2>&1 &
求无穷范数的(最好的)
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ > i.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ > j.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD > k.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD > l.out 2>&1 &

nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ > FEMNIST_NIID_PQ.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD > FEMNIST_NIID_QSGD.out 2>&1 &


nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan PQ > fedprox/CIFAR_IID_PQ.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan PQ > fedprox/CIFAR_NIID_PQ.out 2>&1 &

nohup python -u Server.py --dataset CIFAR-10 --iid IID --quan QSGD > fedprox/CIFAR_IID_QSGD.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-10 --iid NIID --quan QSGD > fedprox/CIFAR_NIID_QSGD.out 2>&1 &

nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan PQ > fedprox/CIFAR100_IID_PQ.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan PQ > fedprox/CIFAR100_NIID_PQ.out 2>&1 &

nohup python -u Server.py --dataset CIFAR-100 --iid IID --quan QSGD > fedprox/CIFAR100_IID_QSGD.out 2>&1 &
nohup python -u Server.py --dataset CIFAR-100 --iid NIID --quan QSGD > fedprox/CIFAR100_NIID_QSGD.out 2>&1 &

nohup python -u Server.py --dataset FEMNIST --iid NIID --quan PQ > fedprox/FEMNIST_NIID_PQ.out 2>&1 &
nohup python -u Server.py --dataset FEMNIST --iid NIID --quan QSGD > fedprox/FEMNIST_NIID_QSGD.out 2>&1 &

