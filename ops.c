#include <omp.h>
#include "tensor.c"

void add_float(tensor t1, tensor t2, tensor t3){
    tensor_meta m1 = t1.meta;
    tensor_meta m2 = t2.meta;
    tensor_meta m3 = t3.meta;


    void* d1 = t1.data;
    void* d2 = t2.data;
    void* d3 = t3.data;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (m3.size + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > m3.size){
            end = m3.size;
        }

        for (int i = start; i < end; i++) {
            int i1 = 0;
            int i2 = 0;

            for(uint8 d=0; d < m3.dim; d++){
                int coord = GET_MIXED_RADIX_DIGIT(i, d, m3._stride, m3.shape);
                i1 += (coord * m1.stride[d]);
                i2 += (coord * m2.stride[d]);
            }

            ((float*)d3)[i] = ((float*)d1)[i1] + ((float*)d2)[i2];
        }
    }

}


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

    return t1;
}
