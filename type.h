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


#define IDX_TO_DIM(i, dim) (dim)-(i)

#define GET_MIXED_RADIX_DIGIT(i, n, strides, shape)  (((i) / (strides)[n]) % (shape)[n])