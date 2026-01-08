rm -rf *.txt
cat ../quan.out | grep "Accu" > quan.txt
cat ../top.out | grep "Accu" > top.txt
cat ../top_quan.out | grep "Accu" > top_quan.txt
cat ../top_diff_quan.out | grep "Accu" > top_diff_quan.txt