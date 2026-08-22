
#include "meta.h"




void meta_free(tensor_meta* meta){
    free(meta->shape);
}


void* meta_alloc(int dim, core_context* ctx){
    size_t size = sizeof(int) * dim * 3 + sizeof(int*);
    return ctx->meta_alloc(size);
}

void meta_stride(int* shape, int* strides, int dim){
    strides[dim-1] = 1;
    for (int8 d = dim-1; d > 0; d--)
        strides[d-1] = shape[d] * strides[d];

}



void meta_set(tensor_meta dest, int* shape, int* stride, int dim){
    int size = dim * sizeof(int);
    dest.ctx->meta_copy(dest.shape, shape, size);
    dest.ctx->meta_copy(dest.stride, stride, size);
    
    meta_stride(shape, dest.__stride, dim);
    dest.dim = dim;
    dest.size = meta_size(shape, dim);
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
