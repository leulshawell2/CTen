#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Column-Major: Left digit (n=0) varies fastest
#define GET_MIXED_RADIX_DIGIT(i, n, strides, shape)  (((i) / (strides)[n]) % (shape)[n])   

#define MAX(a, b)  (a) > (b)? a: b
#define MIN(a, b)  (a) > (b)? b: a
#define True  1
#define False 0

// #define FATAL_ERROR(c, m, a...)  printf(m, a); exit(c);


//takes code, sub-code, tensor
#define ERROR(c, sc, t)  (t)->meta.err = c; (t)->meta.sub_err = sc;


#define OP_ERR  1
#define MEM_ERR 2
#define THREAD_ERR 3 
#define MAJOR_ERR_END 4

#define MALLOC_ERR 1
#define FREE_ERR   2
#define DIM_ERR    3
#define SIZE_ERR   4
#define CONTG_ERR  5
#define SUB_ERR_END 6


//some helpers i don't want to be function calls
#define tensor_isshared(t)  *((t)->meta.ref) != 0
#define tensor_fromother(t) tensor_build((t)->meta.dim, (t)->meta.shape, (t)->meta.e_size, &t, NULL)

                        

typedef unsigned char uint8;
typedef uint8 boolean;
typedef char int8;
typedef unsigned int uint;

typedef struct {
    int* shape;
    int* stride;
    //this is always the contiguous stride. 
    //Never Change. Used for MIxed Radix Count and contiguity check
    int* __stride; 
    uint8 dim;
    uint size;
    uint8 e_size;
    int err;
    int sub_err;
    /**
     * references to the data block; 
     * increamented everytime we build a tensor from the same data block
     */
    
    int* ref; 
} tensor_meta;


/**
 * calculate a stride for a shape
 */
void meta_stride(int* shape, int* strides, int dim);


/**
 * Set the all properties of a tenor meta
 */
void meta_set(tensor_meta dest, int* shape, int* stride, int dim);

/**
 * allocate memory for a tenor_meta block
 * 
 */
void* meta_alloc(int dim);
/**
 * calculate size for shape
 */
int meta_size(int* shape, uint8 dim);

/**
 * print a meta data
 */
void meta_print(tensor_meta* meta);

/**
 * free meta data memory (the shape, stride and __stride)
 */
void meta_free(tensor_meta *meta);
/**
 * calculate the size of a shape
 */
int meta_size(int* shape, uint8 dim);


/**
 * print a meta data
 */
void meta_print(tensor_meta* meta);

