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

typedef struct {
    tensor_meta meta;
    void* data;
    
} tensor;


void meta_stride(int* shape, int* strides, int dim);

/**
 * transopose two dimensions
 * doesn't do copy just shape and stride changes
 */
void tensor_transpose(tensor* t, uint8 dim1, uint8 dim2, tensor* res);

/**
 * permute dims. Unlike transpose this can permute any number of dims
 * @param dims has to be of length t.meta.dim
 */
void tensor_permute(tensor* t, uint8* dims, tensor* res);

/**
 * element count of the tensor
 */
int tensor_size(tensor* t);

/**
 * prints meta data of tensor
 */
void tensor_print_meta(tensor* t);

/**
 * prints the data block of a tensor
 */
void tensor_print_data(tensor* t);

/**
 * build tensor & dim
 * @param dim dimension
 * @param shape shape of tensor int[dim]
 * @param e_size size of a single element (use sizeof)
 * @param pr a pointer to other tensor to share data block. Keep ref count (only safe way to share)
 * @param data a buffer if you already have. Don't share other tensors like this. Use the pr param.
 * 
 * 
 */
void tensor_build(uint8 dim, int* shape, uint8 e_size, tensor* pr, void* data, tensor* res);

/**
 * frees every memory held by a tensor
 */
void tensor_free(tensor* t);


/**
 * create a new contiguous tensor
 * if called on a an already contiguous tensor it will return back the same tensor.
 * so no new mem will be allocated if already contiguous
 */
void tensor_contiguous(tensor* t, tensor* res);

/**
 * does bioligy clone. Every DNS copied
 * new memory, but every other property of input tensor is cloned
 */
void tensor_clone(tensor* t, tensor* res);

/**
 * creates a new tensor (exact but contiguous copy) 
 * it is a call to contiguous
 */
void tensor_copy(tensor* t, tensor* res);


void tensor_shape_copy(tensor t, int* buff);

/**
 * allows viewing a tensor with different shape
 * required contuguous and return non-contiguous (No copy)
 */
void tensor_view(tensor* t, int* shape, int dim, tensor* res);


/**
 * same as view but first calls contiguous if tensor is not contiguous
 */
void tensor_reshape(tensor* t, int* shape, int dim, tensor* res);

/**
 * takes a repeat array and repeats every dimention awith that number
 * {2, 3, 4} means repeat dim 0 twice, dim 1 3 times and dim 2 4 time
 * returns contiguous memory
 */
void tensor_repeat(tensor* t, int* repeat, tensor* res);

/**
 * broadcast and tensor for ops. add new dimas and expand dim sizes of 1
 */
void tensor_broadcast(tensor* t, int* shape, int dim, tensor* res);


/**
 * check if a tensor is contiguous.
 * calculates new strides and compares with the current
 */
boolean tensor_iscontiguous(tensor* t);

/**
 * index a tensor
*/
void tensor_index(tensor* t, int* idxs, tensor* res);
/**
 * clone the meta of a tensor. all pointers taken
 */
void tensor_meta_clone(tensor* dest, tensor* src);

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

