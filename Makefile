TARGETS = tensor.o meta.o cten.o  view_ops.o _view_ops.o
CC = gcc
SRC = ./core/tensor.c ./core/meta.c  ./cten.c ./cpu/view_ops.c ./cpu/impl/_view_ops.c  ./cten.c
SHARED_LIB = libcten

main: ${TARGETS} main.o
	${CC}   -fopenmp ${TARGETS} main.o -o main


main.o: main.c
	gcc -c main.c -fopenmp -o  main.o

${TARGETS}:  ${SRC}
	gcc -fopenmp -c ${SRC}


${SHARED_LIB}:  ${SRC}
	gcc -shared  ${SRC} -o ${SHARED_LIB}.so



clean:
	rm ${TARGETS} main main.o
