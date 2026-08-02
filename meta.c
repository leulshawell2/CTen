#include <stdio.h>
#include "libtensor.h"


int* tensor_shape(tensor t){
    return t.meta.shape;
}


int* tensor_stride(tensor t){
    return t.meta.stride;
}


int tensor_size(tensor t){
    return size(t.meta.shape, t.meta.dim);
}


void tensor_shape_copy(tensor t1, int* shape){

    for(int d=0; d < t1.meta.dim; d++){
        shape[d] = t1.meta.shape[d];
    }

}


void _stride(int* shape, int* strides, int dim){
    strides[0] = 1;
    for (int8 d = 1; d < dim; d++)
        strides[d] = shape[d] * strides[d-1];

}

boolean _array_equal(int* a1, int* a2, int size){
    for(int i =0; i < size; i++){
        if(a1[i] != a2[i])
            return False;
    }
    return True;
}



boolean tensor_iscontiguous(tensor t){
    int con_strides[t.meta.dim];
    _stride(t.meta.shape, con_strides, t.meta.dim);

    return _array_equal(t.meta.stride, con_strides, t.meta.dim);


}


int size(int* shape, uint8 dim){
    int size = 1;
    for(int d=0; d < dim; d++)
        size *= shape[d];

    return size;
}

void tensor_transpose(tensor t, uint8 dim1, uint8 dim2){
    tensor_meta m = t.meta;

    int temp = m.shape[dim1];
    m.shape[dim1] = m.shape[dim2];
    m.shape[dim2] = temp;

    temp = m.stride[dim1];
    m.stride[dim1] = m.stride[dim2];
    m.stride[dim2] = temp;

}

void print_meta(tensor t){
        printf("Size=%d Dim=%d, Shape=[", t.meta.size, t.meta.dim);
        for(int i=0; i<t.meta.dim; i++) printf("%d%s", t.meta.shape[i], i==t.meta.dim-1?"":",");
        printf("], Stride=[");
        for(int i=0; i<t.meta.dim; i++) printf("%d%s", t.meta.stride[i], i==t.meta.dim-1?"":",");
        printf("] Addr=%p\n", t.data);
}
