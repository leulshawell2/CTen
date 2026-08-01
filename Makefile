main: main.o ops.o tensor.o meta.o
	gcc -fopenmp main.o ops.o tensor.o meta.o -o main


main.o ops.o tensor.o meta.o:
	gcc -c main.c ops.c tensor.c meta.c

