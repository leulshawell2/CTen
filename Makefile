TARGETS = tensor.o meta.o
CC = gcc
SRC = tensor.c meta.c
SHARED_LIB = libcten

main: ${TARGETS} main.o
	${CC}   -fopenmp ${TARGETS} main.o -o main


main.o: main.c
	gcc -c main.c -o  main.o

${TARGETS}:  ${SRC}
	gcc -c  main.c  tensor.c meta.c


${SHARED_LIB}:  ${SRC}
	gcc -shared -fopenmp ${SRC} -o ${SHARED_LIB}.so



clean:
	rm ${TARGETS} main main.o
