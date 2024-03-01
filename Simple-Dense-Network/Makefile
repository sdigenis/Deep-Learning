CC=gcc
CFLAGS=-Wall -O3
LDFLAGS=-lm

all:	mlp	kmeans

mlp:	mlp.c	nn_architecture.h	nn_architecture.c	data.h	data.c
	$(CC) $(CFLAGS) -g mlp.c -o mlp $(LDFLAGS)

kmeans:	kmeans.c	data.h	data.c
	$(CC) $(CFLAGS) -g kmeans.c -o kmeans $(LDFLAGS)

clean:
	rm -f mlp kmeans mlp_data.txt kmeans_data.txt
