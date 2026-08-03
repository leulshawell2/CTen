#include <stdio.h>
#include "libtensor.h"




void meta_free(tensor_meta* meta){
    /**the meta data memory starts at shape so we free that  */
    free(meta->shape);
}

void meta_stride(int* shape, int* strides, int dim){
    strides[0] = 1;
    for (int8 d = 1; d < dim; d++)
        strides[d] = shape[d-1] * strides[d-1];
}





void meta_set(tensor_meta dest, int* shape, int* stride, int dim){
    int size = dim * sizeof(int);
    memcpy(dest.shape, shape, size);
    memcpy(dest.stride, stride, size);
    
    meta_stride(shape, dest.__stride, dim);
    dest.dim = dim;
    dest.size = meta_size(shape, dim);


}

boolean _array_equal(int* a1, int* a2, int size){
    for(int i =0; i < size; i++){
        if(a1[i] != a2[i])
            return False;
    }
    return True;
}



int meta_size(int* shape, uint8 dim){
    int size = 1;
    for(int d=0; d < dim; d++)
        size *= shape[d];

    return size;
}


void meta_print(tensor_meta* meta){
        printf("Size=%d Dim=%d, Shape=[",  meta->size,  meta->dim);
        for(int i=0; i< meta->dim; i++) printf("%d%s",  meta->shape[i], i== meta->dim-1?"":",");
        printf("], Stride=[");
        for(int i=0; i< meta->dim; i++) printf("%d%s",  meta->stride[i], i== meta->dim-1?"":",");
        printf("], __Stride=[");
        for(int i=0; i< meta->dim; i++) printf("%d%s",  meta->__stride[i], i== meta->dim-1?"":",");
        printf("] Error=%d Sub-Error=%d\n", meta->err, meta->sub_err);
        
}


void tensor_print_meta(tensor t){
        printf("Addr=%p ", t.data);
        meta_print(&t.meta);

}



boolean tensor_iscontiguous(tensor t){
    return _array_equal(t.meta.stride, t.meta.__stride, t.meta.dim);
}

int* tensor_shape(tensor t){
    return t.meta.shape;
}


int* tensor_stride(tensor t){
    return t.meta.stride;
}


int tensor_size(tensor t){
    return meta_size(t.meta.shape, t.meta.dim);
}


void tensor_shape_copy(tensor t1, int* shape){

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
