#include "_cpu_impl.h"

void _tensor_cast_int32_float32(tensor* t, tensor* res){
        tensor_meta res_meta = res->meta;

        #ifdef OMP
            #pragma omp parallel
            {
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();

                int w_ =  (res_meta.size + nthreads-1) / nthreads;
                int start  = tid * w_;
                int end = start + w_;
                if (end > res_meta.size){
                    end = res_meta.size;
                }
                for (int i = start; i < end; i++) {

        #else
                for (int i = 0; i < res_meta.size; i++) {
        #endif
                    ((float*)res->data)[i] = (float)(((int*)t->data)[i]);
                }
        #ifdef OMP
            }
        #endif

}

void _tensor_cast_float32_int32(tensor* t, tensor* res){
        tensor_meta res_meta = res->meta;

        #ifdef OMP
            #pragma omp parallel
            {
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();

                int w_ =  (res_meta.size + nthreads-1) / nthreads;
                int start  = tid * w_;
                int end = start + w_;
                if (end > res_meta.size){
                    end = res_meta.size;
                }
                for (int i = start; i < end; i++) {

        #else
                for (int i = 0; i < res_meta.size; i++) {
        #endif
                    ((int*)res->data)[i] = (int)(((float*)t->data)[i]);
                }
        #ifdef OMP
            }
        #endif

}
