#include <omp.h>
#include  "libtensor.h"




void tensor_print_data(tensor t){
    tensor_meta m = t.meta;
    
    printf("\n");
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


tensor tensor_build(uint8 dim, int* shape, uint8 e_size, void* data){

    tensor t;
    t.meta.dim = dim;
    t.meta.e_size = e_size;

    int size = 1;
    for(uint8 d=0; d < dim; d++) 
        size = size * shape[d];
    t.meta.size = size;


    int meta_size = sizeof(int) * dim;
    
    void* meta = malloc(meta_size * 3);
    t.meta.shape = meta;
    memcpy(t.meta.shape, shape, meta_size);

    
    
    
    t.meta.stride = meta + meta_size;
    t.meta.__stride = meta + 2 * meta_size;
    
    t.meta.stride[0] = 1;
    for (int8 d = 1; d < dim; d++)
        t.meta.stride[d] = shape[d-1] * t.meta.stride[d-1];


    memcpy(t.meta.__stride, t.meta.stride, meta_size);
    
    if(data){
        t.data = data;
    }else{
        t.data = malloc(size * e_size);
        if(t.data == NULL){
            FATAL_ERROR(1, "MemoryError: error alocating memory size: %d", size * e_size)
        }
    }

    
    return t;
}


void tensor_free(tensor t){
    meta_free(&t.meta);
    free(t.data);
}

tensor tensor_clone(tensor t){
    return  tensor_build(t.meta.dim, t.meta.shape, t.meta.e_size, t.data);
}


tensor tensor_copy(tensor t){
    return tensor_contiguous(t);

}



tensor tensor_contiguous(tensor t){
    
    tensor t1 = tensor_build(t.meta.dim, t.meta.shape, t.meta.e_size, NULL);
    tensor_meta m1 = t1.meta;


    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (m1.size + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > m1.size){
            end = m1.size;
        }

        for (int i = start; i < end; i++) {
            int i1 = 0;
            for(uint8 d=0; d < m1.dim; d++){
                int coord = GET_MIXED_RADIX_DIGIT(i, d, m1.__stride, m1.shape);
                i1 += (coord * t.meta.stride[d]);
            }
            ((float*)t1.data)[i] = ((float*)t.data)[i1];
        }
    }

    return t1;
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
            FATAL_ERROR(1, "ViewError:  tensor size %d -> %d", t.meta.size, size_)
        }
    }

    if(!tensor_iscontiguous(t)){
        FATAL_ERROR(1, "ViewError: tensor is not contiguous", 0)
    }
    
    return tensor_build(dim, shape, t.meta.e_size, t.data);
    
}

tensor tensor_reshape(tensor t, int* shape, int dim){
    if(tensor_iscontiguous(t))
        return tensor_view(t, shape, dim);

    tensor temp = tensor_contiguous(t);
    tensor res = tensor_view(temp, shape, dim);
    meta_free(&temp.meta);

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

    

    tensor temp = tensor_build(temp_dim, shape, t.meta.e_size, t.data);

    temp.meta.stride = strides;
    temp.meta.shape = shape;
    
    temp = tensor_contiguous(temp);

    tensor res = tensor_build(t.meta.dim, new_shape, t.meta.e_size, temp.data);

    meta_free(&temp.meta);

    return res;

}   

tensor tensor_broadcast(tensor t, int* shape, int dim){
    if(dim < t.meta.dim){
        FATAL_ERROR(1, "BroadcastError: dim : %d < %d", dim, t.meta.dim);
    }
    int new_shape[dim];
    int new_stride[dim];

    for(int d=0; d < dim; d++){
        if(d < t.meta.dim){
            if(shape[d] != t.meta.shape[d]){
                if(t.meta.shape[d] != 1){
                    FATAL_ERROR(1, "BroadCastError: can't expand non singular dim: %d. %d -> %d", d, t.meta.shape[d], shape[d]);
                }
                new_stride[d] = 0;
            }else{
                new_stride[d] = t.meta.stride[d];
            }
        }else {
            new_stride[d] = 0;
        }
        new_shape[d] = shape[d];
    }
    
    tensor res = tensor_build(dim, new_shape, t.meta.e_size, t.data);
    
    memcpy(res.meta.stride, new_stride, dim * sizeof(int));

    return res;
}
