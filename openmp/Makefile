CC = gcc
CFLAGS = -fopenmp -O2

.PHONY : all
all : z1 z2 z3 z4 z5

z1 : dz1z1.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z2 : dz1z2.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z3 : dz1z3.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z4 : dz1z4.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z5 : dz1z5.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

pi1 : pi1_manual.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

pi2 : pi2_worksharing.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

pi3 : pi3_tasks.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

needle : needle_parallel.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

nb : nbody_parallel.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

.PHONY : all
clean :
	rm pi1 pi2 pi3 needle nb z1 z2 z3 z4 z5
