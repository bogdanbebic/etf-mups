CC = gcc
CFLAGS = -fopenmp -O2

pi1 : dz1z1.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

pi2 : dz1z2.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

pi3 : dz1z3.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

needle : dz1z4.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

nb : dz1z5.c
	$(CC) $(CFLAGS) -Inbody -o $(@) $(<) -lm

clean :
	rm pi1 pi2 pi3 needle nb
