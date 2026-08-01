main: ops.o tensor.o meta.o
	gcc   -fopenmp main.o ops.o tensor.o meta.o -o main


main.o: main.c
	gcc -c main.c -o  main.o

ops.o tensor.o meta.o: ops.c tensor.c meta.c
	gcc -c  main.c ops.c tensor.c meta.c


libctensor.so: ops.c tensor.c meta.c
	gcc -shared -fopenmp main.c ops.c tensor.c meta.c -o libctensor.so

clean:
	rm *.o main
