
#include  "cten.h"




void tensor_print_data(tensor t){
    tensor_meta m = t.meta;
    
    for (uint i = 0; i < m.size; i++) {
        int i1 = 0;
        for(uint8 d=0; d < m.dim; d++){
            int coord = GET_MIXED_RADIX_DIGIT(i, d, m.__stride, m.shape);
            i1 += (coord * m.stride[d]);
        }
        printf("%f,\t", ((float*)t.data)[i1]);
    }
    printf("\n");

}




tensor tensor_build(uint8 dim, int* shape, uint8 e_size, tensor* pr, void* data){
    
    tensor t;

    
    t.meta.err = 0;
    t.meta.sub_err = 0;
    t.meta.dim = dim;
    t.meta.e_size = e_size;
    
    int size = 1;
    for(uint8 d=0; d < dim; d++) 
        size = size * shape[d];
    t.meta.size = size;
    
    
    int meta_size = sizeof(int) * dim;
    void* meta = meta_alloc(dim);
    if(!meta){
        ERROR(MEM_ERR, MALLOC_ERR, t)
    }
    
    
    //build the meta block    
    memcpy(meta, shape, meta_size);
    t.meta.shape = meta;
    t.meta.stride = meta + meta_size;
    t.meta.__stride = t.meta.stride + dim;
    t.meta.ref   = t.meta.__stride + dim;

    t.meta.stride[0] = shape[0] != 0? 1: 0;
    for (int8 d = 1; d < dim; d++)
        t.meta.stride[d] = shape[d-1] * t.meta.stride[d-1];


    memcpy(t.meta.__stride, t.meta.stride, meta_size);
    
    if(pr){
        //create from other tensor
        *(pr->meta.ref) += 1;
        t.meta.ref = pr->meta.ref;
        t.data = pr->data;
    }else if(data){
        //create from void* data
        t.data = data;
    }else{
        t.data = malloc(size * e_size);
        if(t.data == NULL){
            ERROR(MEM_ERR, MALLOC_ERR, t)
        }
    }

    
    return t;
}


void tensor_free(tensor t){
    if(tensor_isshared(t))
        *(t.meta.ref ) -= 1;
    else
        free(t.data);
    free(t.meta.shape);
}

tensor tensor_clone(tensor t){
    tensor res;
    //copy all the meta in a single memcpy call
    res.meta.shape = meta_alloc(t.meta.dim);

    if(!res.meta.shape){
        ERROR(MEM_ERR, MALLOC_ERR, t);
    }

    res.meta.dim = t.meta.dim;
    res.meta.e_size = t.meta.e_size;
    res.meta.size = t.meta.size;
    res.meta.sub_err = 0;
    res.meta.err = 0;
    res.meta.stride = res.meta.shape + res.meta.dim;
    res.meta.__stride = res.meta.stride + res.meta.dim;
    res.meta.ref   = t.meta.__stride + res.meta.dim; 
    *res.meta.ref = 0;
    
    memcpy(res.meta.shape, t.meta.shape, t.meta.dim * sizeof(int) * 3);
    res.data = malloc(res.meta.size * res.meta.e_size);
    if(!res.data){
        ERROR(MEM_ERR, MALLOC_ERR, t);
    }

    memcpy(res.data, t.data, res.meta.e_size * res.meta.size);
    return res;
}


tensor tensor_copy(tensor t){
    return tensor_contiguous(t);

}


/**
 * No copy to same size
 */
tensor tensor_view(tensor t, int* shape, int dim){

    if(dim==0)
        dim = t.meta.dim;
    else{
        int size_ = meta_size(shape, dim);
        if (size_ != t.meta.size){
            ERROR(OP_ERR, SIZE_ERR, t)
        }
    }

    if(!tensor_iscontiguous(t)){
        ERROR(OP_ERR, CONTG_ERR, t)
    }
    
    return tensor_build(dim, shape, t.meta.e_size, &t, NULL);
    
}

tensor tensor_reshape(tensor t, int* shape, int dim){
    if(tensor_iscontiguous(t))
        return tensor_view(t, shape, dim);

    tensor temp = tensor_contiguous(t);
    tensor res = tensor_view(temp, shape, dim);
    tensor_free(temp);

    return res;
}


tensor tensor_repeat(tensor t, int* repeat){
    int temp_dim = t.meta.dim * 2;
    int shape[temp_dim];
    int strides[temp_dim];
    int new_shape[t.meta.dim];

    int d_2;
    for(uint8 d=0; d < temp_dim; d += 2){
        d_2 = d/2;
        shape[d] = repeat[d_2];
        shape[d+1] = t.meta.shape[d_2];

        strides[d] = 0;
        strides[d+1] = t.meta.stride[d_2];
        new_shape[d_2] = t.meta.shape[d_2] * repeat[d_2];
        

    }

    

    tensor temp = tensor_build(temp_dim, shape, t.meta.e_size, &t, NULL);

    temp.meta.stride = strides;
    temp.meta.shape = shape;
    
    temp = tensor_contiguous(temp);

    tensor res = tensor_build(t.meta.dim, new_shape, t.meta.e_size, &temp, NULL);

    tensor_free(temp);

    return res;

}   

tensor tensor_broadcast(tensor t, int* shape, int dim){
    if(dim < t.meta.dim){
        ERROR(OP_ERR, DIM_ERR, t)
    }
    int new_shape[dim];
    int new_stride[dim];

    for(int d=0; d < dim; d++){
        if(d < t.meta.dim){
            if(shape[d] != t.meta.shape[d]){
                if(t.meta.shape[d] != 1 && shape[d] != 1){
                    ERROR(OP_ERR, DIM_ERR, t)
                }
                new_stride[d] = 0;
            }else{
                new_stride[d] = t.meta.stride[d];
            }
        }else {
            new_stride[d] = 0;
        }
        new_shape[d] = shape[d] != 1? shape[d]: t.meta.shape[d];
    }
    
    tensor res = tensor_build(dim, new_shape, t.meta.e_size, &t, NULL);
    
    memcpy(res.meta.stride, new_stride, dim * sizeof(int));

    return res;
}


tensor tensor_index(tensor t, int* idxs){
    int new_shape[t.meta.dim];
    int offset = 0;
    int new_strides[t.meta.dim];

    for(int d=0; d < t.meta.dim * 3;  d+=3 ){
        int s = idxs[d];
        int e = idxs[d + 1];
        int j = idxs[d+2];

        int d_3 = d/3;
        
        new_shape[d_3] = MAX(0, (e - s - 1)/ j + 1);
        new_strides[d_3] = j * t.meta.stride[d_3];

        offset += t.meta.stride[d_3] * s;
    }

    tensor temp = tensor_build(t.meta.dim, new_shape, t.meta.e_size, &t, NULL);
    memcpy(temp.meta.stride, new_strides, t.meta.dim * sizeof(int));

    temp.data += offset * temp.meta.e_size;

    tensor res = tensor_contiguous(temp);
    
    tensor_free(temp);
    
    return res;

}
