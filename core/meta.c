#include <stdio.h>
#include "cten.h"




void meta_free(tensor_meta* meta){
    free(meta->shape);
}



void* meta_alloc(int dim){
    size_t size = sizeof(int) * dim * 3 + sizeof(int*);
    return malloc(size);
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
        printf("Size=%d Dim=%d, Ref=%d Shape=[",  meta->size,  meta->dim, *meta->ref);
        for(int i=0; i< meta->dim; i++) printf("%d%s",  meta->shape[i], i== meta->dim-1?"":",");
        printf("], Stride=[");
        for(int i=0; i< meta->dim; i++) printf("%d%s",  meta->stride[i], i== meta->dim-1?"":",");
        printf("], __Stride=[");
        for(int i=0; i< meta->dim; i++) printf("%d%s",  meta->__stride[i], i== meta->dim-1?"":",");
        printf("] Error=%d Sub-Error=%d\n", meta->err, meta->sub_err);
        
}


void tensor_print_meta(tensor* t){
        printf("Addr=%p ", t->data);
        meta_print(&t->meta);

}



boolean tensor_iscontiguous(tensor* t){
    return _array_equal(t->meta.stride, t->meta.__stride, t->meta.dim);
}


int tensor_size(tensor* t){
    return meta_size(t->meta.shape, t->meta.dim);
}




void tensor_meta_clone(tensor* dest, tensor* src){
    dest->meta.dim = src->meta.dim;
    dest->meta.e_size = src->meta.e_size;
    dest->meta.size = src->meta.size;
    dest->meta.sub_err = src->meta.sub_err;
    dest->meta.err = src->meta.err;
    memcpy(dest->meta.shape, dest->meta.shape, sizeof(int) * src->meta.dim * 3 + sizeof(int*));
}