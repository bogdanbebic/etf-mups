CC = nvcc
CFLAGS =

.PHONY : all
all : z1 z2 z3

z1 : dz4z1.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z2 : dz4z2.c
	$(CC) $(CFLAGS) -o $(@) $(<)

z3 : dz4z3.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

pi : pi_parallel.cu
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

needle : needle_parallel.cu
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

nb : nbody_parallel.cu
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

.PHONY : clean
clean :
	rm pi needle nb z1 z2 z3
