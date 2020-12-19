CC = mpicc
CFLAGS = -O2

.PHONY : all
all : z1 z2 z3 z4

z1 : dz2z1.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z2 : dz2z2.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

z3 : dz2z3.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

z4 : dz2z4.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

pi : pi_parallel.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

needle : needle_parallel.c
	$(CC) $(CFLAGS) -o $(@) $(<) -lm

nb : nbody_parallel.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

nbmw : nbody_mw_parallel.c
	$(CC) $(CFLAGS) -Wno-unused-result -Inbody -o $(@) $(<) -lm

.PHONY : all
clean :
	rm pi needle nb nbmw z1 z2 z3 z4
