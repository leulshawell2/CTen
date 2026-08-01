#include <omp.h>
#include "ctensor.h"


/**
 * create a new contiguous tensor
 */
tensor tensor_contiguous(tensor t){
    
    tensor t1 = build_tensor(t.meta.dim, t.meta.shape, t.meta.e_size, NULL);
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

    printf("%p\n", t.data);
    printf("%p\n", t1.data);

    return t1;
}


/**
 * return a new tensor that points to the same data block
 */
tensor tensor_clone(tensor t){
    return  build_tensor(t.meta.dim, t.meta.shape, t.meta.e_size, t.data);
}

/**
 * creates a new tensor (exact but contiguous copy) 
 * it is a call to contiguous
 */
tensor tensor_copy(tensor t){
    return tensor_contiguous(t);

}
