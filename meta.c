#include <stdio.h>
#include "ctensor.h"


int* tensor_shape(tensor t){
    return t.meta.shape;
}


int* tensor_stride(tensor t){
    return t.meta.stride;
}


void shape_copy(tensor t1, int* shape){

    for(int d=0; d < t1.meta.dim; d++){
        shape[d] = t1.meta.shape[d];
    }

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
        printf("]\n");
}
