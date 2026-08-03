#include  <stdio.h>
#include <stdlib.h>
#include <string.h>

// Column-Major: Left digit (n=0) varies fastest
#define GET_MIXED_RADIX_DIGIT(i, n, strides, shape)  (((i) / (strides)[n]) % (shape)[n])   

#define MAX(a, b)  (a) > (b)? a: b
#define MIN(a, b)  (a) > (b)? b: a
#define True  1
#define False 0

#define FATAL_ERROR(c, msg, args...)  printf(msg, args); exit(c);
                                

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
} tensor_meta;

typedef struct {
    tensor_meta meta;
    void* data;
} tensor;


void meta_stride(int* shape, int* strides, int dim);

/**
 * get shape of a tensor
 */
int* tensor_shape(tensor t);

/**
 * get the stride of a tensor
 */
int* tensor_stride(tensor t);

/**
 * transopose two dimensions
 * doesn't do copy just shape and stride changes
 */
void tensor_transpose(tensor t, uint8 dim1, uint8 dim2);

/**
 * element count of the tensor
 */
int tensor_size(tensor t);

/**
 * prints meta data of tensor
 */
void tensor_print_meta(tensor t);

/**
 * prints the data block of a tensor
 */
void tensor_print_data(tensor t);

/**
 * build tensor & dim
 * @param dim: dimension
 * @param shape: shape of tensor int[dim]
 * @param e_size: size of a single element (use sizeof)
 * @param data: a data block from other tensor if you don't want new mem allocated 
 * 
 * 
 */
tensor tensor_build(uint8 dim, int* shape, uint8 e_size, void* data);

/**
 * frees every memory held by a tensor
 */
void tensor_free(tensor t);


/**
 * create a new contiguous tensor
 */
tensor tensor_contiguous(tensor t);

/**
 * return a new tensor that points to the same data block
 */
tensor tensor_clone(tensor t);

/**
 * creates a new tensor (exact but contiguous copy) 
 * it is a call to contiguous
 */
tensor tensor_copy(tensor t);


void tensor_shape_copy(tensor t, int* buff);

/**
 * allows viewing a tensor with different shape
 * required contuguous and return non-contiguous (No copy)
 */
tensor tensor_view(tensor t, int* shape, int dim);


/**
 * same as view but first calls contiguous if tensor is not contiguous
 * 
 */
tensor tensor_reshape(tensor t, int* shape, int dim);

/**
 * takes a repeat array and repeats every dimention awith that number
 * {2, 3, 4} means repeat dim 0 twice, dim 1 3 times and dim 2 4 time
 * returns contiguous memory
 */
tensor tensor_repeat(tensor t, int* repeat);

int size(int* shape, uint8 dim);

/**
 * check if a tensor is contiguous.
 * calculates new strides and compares with the current
 */
boolean tensor_iscontiguous(tensor t);

/**
 * Set the all properties of a tenor meta
 */
void meta_set(tensor_meta dest, int* shape, int* stride, int dim);

/**
 * print a meta data
 */
void meta_print(tensor_meta* meta);

/**
 * free meta data memory (the shape, stride and __stride)
 */
void meta_free(tensor_meta *meta);