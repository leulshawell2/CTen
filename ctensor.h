#include  <stdio.h>
#define GET_MIXED_RADIX_DIGIT(i, n, strides, shape)  (((i) / (strides)[n]) % (shape)[n])



typedef unsigned char uint8;
typedef char int8;
typedef unsigned int uint;

typedef struct {
    int* shape;
    int* stride;
    int* _stride;
    uint8 dim;
    uint size;
    uint8 e_size;
} tensor_meta;

typedef struct {
    tensor_meta meta;
    void* data;
} tensor;


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
int size(tensor t);

/**
 * prints meta data of tensor
 */
void print_meta(tensor t);

/**
 * prints the data block of a tensor
 */
void print_data(tensor t);

/**
 * build tensor & dim
 * @param dim: dimension
 * @param shape: shape of tensor int[dim]
 * @param e_size: size of a single element (use sizeof)
 * @param data: a data block from other tensor if you don't want new mem allocated 
 * 
 * 
 */
tensor build_tensor(uint8 dim, int* shape, uint8 e_size, void* data);

/**
 * frees every memory held by a tensor
 */
void free_tensor(tensor t);


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

tensor concat_new_dim(tensor t1, tensor t2);


void shape_copy(tensor t1, int* buff);