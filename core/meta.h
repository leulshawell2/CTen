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
#define ERROR(c, sc, t)  (t)->meta.err = c; (t)->meta.sub_err = sc; return;


#define OP_ERR  1
#define MEM_ERR 2
#define THREAD_ERR 3 
#define CORE_MAJOR_ERR_END 4

#define MALLOC_ERR 1
#define FREE_ERR   2
#define DIM_ERR    3
#define SIZE_ERR   4
#define CONTG_ERR  5
#define CORE_SUB_ERR_END 6


typedef unsigned char uint8;
typedef uint8 boolean;
typedef char int8;
typedef unsigned int uint;




typedef void*(*alloc_func)(size_t);
typedef void * (*copy_func)(void * restrict,  const void * restrict,  size_t);
typedef int (*cmp_func)(const void *, const void*, size_t);
typedef void(*free_func)(void*);



typedef struct {
    //allocate memory block
    alloc_func data_alloc;
    alloc_func meta_alloc;
    //copy memory block
    copy_func data_copy;
    copy_func meta_copy;
    //compare memory blocks
    cmp_func meta_cmp;
    cmp_func data_cmp;
    //free a memory block
    free_func meta_free;
    free_func data_free;
}core_context;




enum dtypes {
    None=0,
    Int32,
    Int16,
    Int8,
    Float32,
    Float16,
};


typedef struct {
    int* shape;
    int* stride;
    //this is always the contiguous stride. Never Change. Used for MIxed Radix Count and contiguity check
    int* __stride; 

    uint8 dim;
    uint size;
    uint8 e_size;

    int err;
    int sub_err;
    uint8 dtype;
    /**
     * references to the data block; 
     * increamented everytime we build a tensor from the same data block
     */
    int* ref; 

    core_context* ctx;
} tensor_meta;


/**
 * calculate a stride for a shape
 */
void meta_stride(int*, int*, int);


/**
 * Set the all properties of a tenor meta
 */
void meta_set(tensor_meta, int*, int*, int);

/**
 * allocate memory for a tenor_meta block
 * 
 */
void* meta_alloc(int, core_context*);
/**
 * calculate size for shape
 */
int meta_size(int*, uint8);

/**
 * print a meta data
 */
void meta_print(tensor_meta*);

/**
 * free meta data memory (the, stride and __stride)
 */
void meta_free(tensor_meta*);
/**
 * calculate the size of a shape
 */
int meta_size(int*, uint8);

/**
 * print a meta data
 */
void meta_print(tensor_meta*);

