#include "meta.h"



typedef struct {
    tensor_meta meta;
    void* data;
} tensor;




/**
 * this is a macor that creates a tensor and sets the ctx
 * @param t name of vatiable
 * @param ctx cten core_context
 */

#define tensor_from_ctx(t, c) tensor t; t.meta.ctx = c;






/**
 * build tensor & dim
 * @param dim dimension
 * @param shape shape of tensor int[dim]
 * @param e_size size of a single element (use sizeof)
 * @param dtype what is the data type
 * @param pr a pointer to other tensor to share data block. Keep ref count (only safe way to share)
 * @param data a buffer if you already have. Don't share other tensors like this. Use the pr param.
 * 
 * 
 */
void tensor_build(uint8 dim, int* shape, uint8 e_size, uint8 dtype, tensor* pr, void* data, tensor* res);

/**
 * frees every memory held by a tensor
 * if data  block is shared it just decrements the shared ref
 */
void tensor_free(tensor* t);


/**
 * print the meta data of a tensor
 */
void tensor_print_meta(tensor* t);

/**
 * check if a tensor is contiguous
 */
boolean tensor_iscontiguous(tensor* t);

/**
 * creates a tensor from a scalar
 * it is a 0 stride walk of a scalar
 */

 void tensor_from_scalar(int dim, int* shape, int e_size, int dtype, void* data, tensor* res);


/**
 * clone the meta of a tenor to ther
 */
void tensor_meta_clone(tensor* dest, tensor* src);

/**
 * print a tensor data
 */
void tensor_print_data(tensor* t);

//heloerps for not handling context setting manually
void tensors_of_context(tensor* ts, int count, core_context* ctx);