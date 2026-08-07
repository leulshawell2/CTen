CC = gcc
SHARED_LIB = libcten



TARGETS = tensor.o \
		meta.o \
		cten.o  \
		view.o \
		_view.o\
		movement.o \
		_movement.o\
		binary.o \
		_binary.o 


SRC = ./core/tensor.c \
	 ./core/meta.c \
	 ./cten.c \
	 ./backend/cpu/view.c \
	 ./backend/cpu/movement.c \
	 ./backend/cpu/unary.c \
	 ./backend/cpu/binary.c \
	 ./backend/cpu/impl/_view.c \
	 ./backend/cpu/impl/_movement.c \
	 ./backend/cpu/impl/_unary.c \
	 ./backend/cpu/impl/_binary.c \
	./cten.c


main: ${TARGETS} main.o
	${CC}   -fopenmp ${TARGETS} main.o -o main


main.o: main.c
	gcc -c main.c -o  main.o

${TARGETS}:  ${SRC}
	gcc -c ${SRC}


${SHARED_LIB}:  ${SRC}
	gcc -shared  ${SRC} -o ${SHARED_LIB}.so


clean:
	rm ${TARGETS} main main.o
