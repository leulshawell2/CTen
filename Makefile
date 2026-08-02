main: tensor.o meta.o
	gcc   -fopenmp main.o tensor.o meta.o -o main


main.o: main.c
	gcc -c main.c -o  main.o

ops.o tensor.o meta.o:  tensor.c meta.c
	gcc -c  main.c  tensor.c meta.c


libctensor.so:  tensor.c meta.c
	gcc -shared -fopenmp main.c  tensor.c meta.c -o libctensor.so

clean:
	rm *.o main
