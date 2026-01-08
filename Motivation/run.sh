rm -rf *.out
nohup python -u Server.py --fit 1 > fit.out 2>&1 &
nohup python -u Server.py --fit 0 > top.out 2>&1 &
