OBJS = test_host test_dev0 test_dev1 test_arch test_dev2
SCRIPT = run.sh

all: $(OBJS) $(SCRIPT)

$(SCRIPT): $(OBJS)
	@echo "#!/bin/bash" > $(SCRIPT)
	@for i in $(OBJS); do echo "echo -n '$$i '" >> $(SCRIPT); echo "./$$i" >> $(SCRIPT); done
	@echo "----------"
	@echo created $(SCRIPT)

test_host: test.c
	gcc -o $@ $<

test_dev0: test0.cu
	nvcc -o $@ $<

test_dev1: test1.cu
	nvcc -o $@ $<

test_arch: test1.cu
	nvcc -o $@ $< -arch=sm_61

test_dev2: test2.cu
	nvcc -o $@ $<

clean:
	rm -f $(OBJS) $(SCRIPT)
