#include <stdio.h>
#include "cten.h"




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
        printf("Size=%d Dim=%d, Ref=%d Shape=[",  meta->size,  meta->dim, *meta->ref);
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


int tensor_size(tensor t){
    return meta_size(t.meta.shape, t.meta.dim);
}


void tensor_shape_copy(tensor t1, int* shape){

    for(int d=0; d < t1.meta.dim; d++){
        shape[d] = t1.meta.shape[d];
    }

}



tensor tensor_transpose(tensor t, uint8 dim1, uint8 dim2){
    tensor_meta m = t.meta;

    int new_shape[m.dim];

    memcpy(new_shape, m.shape, m.dim * sizeof(int));

    int temp = new_shape[dim1];
    new_shape[dim1] = new_shape[dim2];
    new_shape[dim2] = temp;
    
    
    tensor res = tensor_build(m.dim, new_shape, m.e_size, &t, NULL);

    memcpy(res.meta.stride, m.stride, m.dim * sizeof(int));

    temp = m.stride[dim1];
    res.meta.stride[dim1] = res.meta.stride[dim2];
    res.meta.stride[dim2] = temp;


    return res;

}


tensor tensor_permute(tensor t, uint8* dims){
    tensor_meta m = t.meta;

    int new_shape[m.dim];
    int new_strides[m.dim];
    int temp = 0;
    int d2;

    memcpy(new_shape, m.shape, m.dim * sizeof(int));
    for(uint8 d=0; d < t.meta.dim; d++){
        d2 = dims[d];
        new_shape[d] = m.shape[d2];
        new_strides[d] = m.stride[d2];
    }
    
    tensor res = tensor_build(m.dim, new_shape, m.e_size, &t, NULL);
    memcpy(res.meta.stride, new_strides, m.dim * sizeof(int));

    return res;

}



