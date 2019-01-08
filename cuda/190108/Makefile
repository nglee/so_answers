all:
	gcc -o test_host test.c
	nvcc -o test_dev0 test0.cu
	nvcc -o test_dev1 test1.cu
	nvcc -o test_dev1_arch -arch=sm_61 test1.cu

clean:
	rm -f test_host test_dev0 test_dev1 test_dev1_arch
