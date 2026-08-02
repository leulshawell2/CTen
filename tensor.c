#include <omp.h>
#include  "libtensor.h"



void print_data(tensor t){
    tensor_meta m = t.meta;
    
    printf("\n");
    for (uint i = 0; i < m.size; i++) {
        int i1 = 0;

        for(uint8 d=0; d < m.dim; d++){
            int coord = GET_MIXED_RADIX_DIGIT(i, d, m._stride, m.shape);
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
    t.meta._stride = meta + 2 * meta_size;

    t.meta.stride[0] = 1;
    for (int8 d = 1; d < dim; d++)
        t.meta.stride[d] = shape[d] * t.meta.stride[d-1];

    memcpy(t.meta._stride, t.meta.stride, meta_size);
    
    if(data){
        t.data = data;
    }else{
        t.data = malloc(size * e_size);
        if(t.data == NULL){
            printf("Error allocating memory. size: %d", size * e_size);
            exit(1);
        }
    }

    for(int i=0; i < size; i++){
        ((float*)t.data)[i] = i;
    }

    return t;
}


void tensor_free(tensor t){
    free(t.meta.shape);
    free((void*)t.data);
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
                int coord = GET_MIXED_RADIX_DIGIT(i, d, m1._stride, m1.shape);
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
tensor tensor_view(tensor t1, int* shape, int dim){

    if(dim==0)
        dim = t1.meta.dim;
    else{
        int size_ = size(shape, dim);
        if (size_ != t1.meta.size){
            FATAL_ERROR(1, "ViewError:  tensor size %d -> %d", t1.meta.size, size_)
        }
    }

    if(!tensor_iscontiguous(t1)){
        FATAL_ERROR(1, "ViewError: tensor is not contiguous", 0)
    }
    
    return tensor_build(dim, shape, t1.meta.e_size, t1.data);
    
}

/**
 * No copy to higher dim
 */
tensor tensor_broadcast(tensor t1, int* shape, int dim){
    
}

/**
 * Copies data
 */
tensor tensor_repeat(tensor t1, int* shape, int dim){
    
}