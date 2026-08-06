
#include  "cten.h"




void tensor_print_data(tensor* t){
    tensor_meta m = t->meta;
    
    for (uint i = 0; i < m.size; i++) {
        int i1 = 0;
        for(uint8 d=0; d < m.dim; d++){
            int coord = GET_MIXED_RADIX_DIGIT(i, d, m.__stride, m.shape);
            i1 += (coord * m.stride[d]);
        }
        printf("%f,\t", ((float*)t->data)[i1]);
    }
    printf("\n");

}




void tensor_build(uint8 dim, int* shape, uint8 e_size, tensor* pr, void* data, tensor* res){
    
    res->meta.err = 0;
    res->meta.sub_err = 0;
    res->meta.dim = dim;
    res->meta.e_size = e_size;
    
    int size = 1;
    for(uint8 d=0; d < dim; d++) 
        size = size * shape[d];
    res->meta.size = size;
    
    
    int meta_size = sizeof(int) * dim;
    void* meta = meta_alloc(dim);
    if(!meta){
        ERROR(MEM_ERR, MALLOC_ERR, res)
    }
    
    
    //build the meta block    
    memcpy(meta, shape, meta_size);
    res->meta.shape = meta;
    res->meta.stride = meta + meta_size;
    res->meta.__stride = res->meta.stride + dim;
    res->meta.ref   = res->meta.__stride + dim;

    res->meta.stride[0] = shape[0] != 0? 1: 0;
    for (int8 d = 1; d < dim; d++)
        res->meta.stride[d] = shape[d-1] * res->meta.stride[d-1];


    memcpy(res->meta.__stride, res->meta.stride, meta_size);
    
    if(pr){
        //create from other tensor
        *(pr->meta.ref) += 1;
        res->meta.ref = pr->meta.ref;
        res->data = pr->data;
    }else if(data){
        //create from void* data
        res->data = data;
    }else{
        res->data = malloc(size * e_size);
        if(res->data == NULL){
            ERROR(MEM_ERR, MALLOC_ERR, res)
        }
    }
    
}


void tensor_free(tensor* t){
    if(tensor_isshared(t))
        *(t->meta.ref ) -= 1;
    else
        free(t->data);
    free(t->meta.shape);
}

void tensor_clone(tensor* t, tensor* res){
    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, NULL, NULL, res);
    memcpy(res->meta.stride, t->meta.stride, t->meta.dim * sizeof(int));
    memcpy(res->data, t->data, res->meta.e_size * res->meta.size);
}


void tensor_copy(tensor* t, tensor* res){
    return tensor_contiguous(t, res);

}

void tensor_view(tensor* t, int* shape, int dim, tensor* res){

    if(dim==0)
        dim = t->meta.dim;
    else{
        int size_ = meta_size(shape, dim);
        if (size_ != t->meta.size){
            ERROR(OP_ERR, SIZE_ERR, res)
        }
    }

    if(!tensor_iscontiguous(t)){
        ERROR(OP_ERR, CONTG_ERR, res)
    }
    tensor_build(dim, shape, t->meta.e_size, t, NULL, res);
}

void tensor_reshape(tensor* t, int* shape, int dim, tensor* res){
    if(tensor_iscontiguous(t))
        tensor_view(t, shape, dim, res);

    tensor temp[1];
    tensor_contiguous(t, temp);
    tensor_view(temp, shape, dim, res);

}


void tensor_repeat(tensor* t, int* repeat, tensor* res){
    int temp_dim = t->meta.dim * 2;
    int shape[temp_dim];
    int strides[temp_dim];
    int new_shape[t->meta.dim];

    int d_2;
    for(uint8 d=0; d < temp_dim; d += 2){
        d_2 = d/2;
        shape[d] = repeat[d_2];
        shape[d+1] = t->meta.shape[d_2];

        strides[d] = 0;
        strides[d+1] = t->meta.stride[d_2];
        new_shape[d_2] = t->meta.shape[d_2] * repeat[d_2];

    }
    tensor temps[2];
    tensor_build(temp_dim, shape, t->meta.e_size, t, NULL, temps);
    temps->meta.stride = strides;

    tensor_build(temp_dim, shape, t->meta.e_size, NULL, NULL, temps + 1);
    tensor_contiguous(temps, temps + 1);
    tensor_build(t->meta.dim, new_shape, t->meta.e_size, temps + 1, NULL, res);

    tensor_free(temps);
    tensor_free(temps + 1);
}   

void tensor_broadcast(tensor* t, int* shape, int dim, tensor* res){
    if(dim < t->meta.dim){
        ERROR(OP_ERR, DIM_ERR, t)
    }
    int new_shape[dim];
    int new_stride[dim];

    for(int d=0; d < dim; d++){
        if(d < t->meta.dim){
            if(shape[d] != t->meta.shape[d]){
                if(t->meta.shape[d] != 1 && shape[d] != 1){
                    ERROR(OP_ERR, DIM_ERR, t)
                }
                new_stride[d] = 0;
            }else{
                new_stride[d] = t->meta.stride[d];
            }
        }else {
            new_stride[d] = 0;
        }
        new_shape[d] = shape[d] != 1? shape[d]: t->meta.shape[d];
    }

    tensor* temp;
    tensor_build(dim, new_shape, t->meta.e_size, t, NULL, temp);
    tensor_meta_clone(res, temp);
    res->data = temp->data;
    
    memcpy(res->meta.stride, new_stride, dim * sizeof(int));

}


void tensor_index(tensor* t, int* idxs, tensor* res){
    int new_shape[t->meta.dim];
    int offset = 0;
    int new_strides[t->meta.dim];

    for(int d=0; d < t->meta.dim * 3;  d+=3 ){
        int s = idxs[d];
        int e = idxs[d + 1];
        int j = idxs[d+2];

        int d_3 = d/3;
        
        new_shape[d_3] = MAX(0, (e - s - 1)/ j + 1);
        new_strides[d_3] = j * t->meta.stride[d_3];

        offset += t->meta.stride[d_3] * s;
    }
    tensor temp_;
    tensor* temp = &temp_;
    tensor_build(t->meta.dim, new_shape, t->meta.e_size, t, NULL, temp);

    temp->meta.stride = new_strides;
    temp->data += offset * temp->meta.e_size;

    tensor_contiguous(temp, res);
    tensor_free(temp);
}

void tensor_transpose(tensor* t, uint8 dim1, uint8 dim2, tensor* res){
    tensor_meta m = t->meta;
    
    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t, NULL, res);
    res->meta.stride = t->meta.stride;
    
    int tmp = res->meta.shape[dim1];
    res->meta.shape[dim1] = res->meta.shape[dim2];
    res->meta.shape[dim2] = tmp;
    
    
    tmp = m.stride[dim1];
    res->meta.stride[dim1] = m.stride[dim2];
    res->meta.stride[dim2] = tmp;

}


void tensor_permute(tensor* t, uint8* dims, tensor* res){
    tensor_meta m = t->meta;

    int new_shape[m.dim];
    int new_strides[m.dim];
    int d2;

    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t, NULL, res);
    res->meta.stride = t->meta.stride;

    memcpy(new_shape, m.shape, m.dim * sizeof(int));
    for(uint8 d=0; d < t->meta.dim; d++){
        d2 = dims[d];
        res->meta.shape[d] = m.shape[d2];
        res->meta.stride[d] = m.stride[d2];
    }
    

    res->data = t->data;
    *res->meta.ref += 1;
}

