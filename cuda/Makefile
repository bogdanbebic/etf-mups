CC = nvcc
CFLAGS =

.PHONY : all
all : z1 z2 z3

z1 : dz4z1.cu
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z2 : dz4z2.cu
	$(CC) $(CFLAGS) -o $(@) $(<)

z3 : dz4z3.cu
	$(CC) $(CFLAGS) -Inbody -o $(@) $(<) -lm

pi : pi_parallel.cu
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

needle : needle_parallel.cu
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

nb : nbody_parallel.cu
	$(CC) $(CFLAGS) -Inbody -o $(@) $(<) -lm

.PHONY : clean
clean :
	rm pi needle nb z1 z2 z3
