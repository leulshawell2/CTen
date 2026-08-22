#include "../../../cten.h"

typedef void(*mapper)(void*, void*);


#ifdef OMP
#include <omp.h>
#endif



/**
 * does bioligy clone. Every DNS copied
 * new memory, but every other property of input tensor is cloned
 */
void _tensor_clone(tensor* t, tensor* res);

/**
 * create a new contiguous tensor
 * if called on a an already contiguous tensor it will return back the same tensor.
 * so no new mem will be allocated if already contiguous
 */
void _tensor_contiguous(tensor* t, tensor* res);

/**
 * creates a new tensor (exact but contiguous copy) 
 * it is a call to contiguous
 */
void _tensor_copy(tensor* t, tensor* res);


/**
 * allows viewing a tensor with different shape
 * required contuguous and return non-contiguous (No copy)
 */
void _tensor_view(tensor* t, int* shape, int dim, tensor* res);


/**
 * same as view but first calls contiguous if tensor is not contiguous
 */
void _tensor_reshape(tensor* t, int* shape, int dim, tensor* res);

/**
 * takes a repeat array and repeats every dimention awith that number
 * {2, 3, 4} means repeat dim 0 twice, dim 1 3 times and dim 2 4 time
 * returns contiguous memory
 */
void _tensor_repeat(tensor* t, int* repeat, tensor* res);

/**
 * broadcast and tensor for ops. add new dimas and expand dim sizes of 1
 */
void _tensor_broadcast(tensor* t, int* shape, int dim, tensor* res);


/**
 * check if a tensor is contiguous.
 * calculates new strides and compares with the current
 */
boolean tensor_iscontiguous(tensor* t);

/**
 * index a tensor
*/
void _tensor_index(tensor* t, int* idxs, tensor* res);
/**
 * clone the meta of a tensor. all pointers taken
 */
void _tensor_meta_clone(tensor* dest, tensor* src);

/**
 * transpose two dims
 */
void _tensor_transpose(tensor* t, uint8 dim1, uint8 dim2, tensor* res);

/**
 * permute dims.void _tensor_permute(tensor* t, uint8* dims, tensor* res){
 */
void _tensor_permute(tensor* t, uint8* dims, tensor* res);

/**
 * helper for matching shapes of 2 tensors
 * assuming the two shapes meet the requirements for broadcasting
 */
void _tensor_broadcast_match(tensor* _t1, tensor* _t2, tensor* t1, tensor* t2 );

/**
 * naive math op kernels for the cpu =============================
 */


void _tensor_add_float32(tensor* t1, tensor* t2, tensor* res);
void _tensor_add_int32(tensor* t1, tensor* t2, tensor* res);

void _tensor_sub_float32(tensor* t1, tensor* t2, tensor* res);
void _tensor_sub_int32(tensor* t1, tensor* t2, tensor* res);

void _tensor_matmul_float32(tensor* t1, tensor* t2, tensor* res);
void _tensor_matmul_int32(tensor* t1, tensor* t2, tensor* res);


void _tensor_div_float32(tensor* t1, tensor* t2, tensor* res);

void _tensor_cast_int32_float32(tensor* t, tensor* res);
void _tensor_cast_float32_int32(tensor* t, tensor* res);


void _tensor_map(tensor* t,  mapper map, tensor* res);

void _tensor_max_f32(tensor* t, tensor* res);
void _tensor_min_f32(tensor* t, tensor* res);