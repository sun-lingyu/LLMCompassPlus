rm *.csv
rm *.pdf

cd ../..

python -m test.matmul.test_matmul --simgpu --roofline

python -m test.matmul.test_matmul --simgpu

cd matmul
python plot_matmul.py