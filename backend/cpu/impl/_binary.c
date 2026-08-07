
#include  "_cpu_impl.h"



void _tensor_add_float32(tensor* t1, tensor* t2, tensor* res){
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
                    int i1 = 0;
                    int i2 = 0;
                    for(uint8 d=0; d < res_meta.dim; d++){
                        int coord1 = GET_MIXED_RADIX_DIGIT(i, d, t1->meta.__stride, t1->meta.shape);
                        int coord2 = GET_MIXED_RADIX_DIGIT(i, d, t2->meta.__stride, t2->meta.shape);
                        i1 += (coord1 * t1->meta.stride[d]);
                        i2 += (coord2 * t2->meta.stride[d]);
                    }
                    ((float*)res->data)[i] = ((float*)t1->data)[i1] + ((float*)t2->data)[i2];
                }
        #ifdef OMP
            }
        #endif

}

void _tensor_add_int32(tensor* t1, tensor* t2, tensor* res){
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
                    int i1 = 0;
                    int i2 = 0;
                    for(uint8 d=0; d < res_meta.dim; d++){
                        int coord1 = GET_MIXED_RADIX_DIGIT(i, d, t1->meta.__stride, t1->meta.shape);
                        int coord2 = GET_MIXED_RADIX_DIGIT(i, d, t2->meta.__stride, t2->meta.shape);
                        i1 += (coord1 * t1->meta.stride[d]);
                        i2 += (coord2 * t2->meta.stride[d]);
                    }
                    ((int*)res->data)[i] = ((int*)t1->data)[i1] + ((int*)t2->data)[i2];
                }
        #ifdef OMP
            }
        #endif

}


void _tensor_sub_float32(tensor* t1, tensor* t2, tensor* res){
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
                    int i1 = 0;
                    int i2 = 0;
                    for(uint8 d=0; d < res_meta.dim; d++){
                        int coord1 = GET_MIXED_RADIX_DIGIT(i, d, t1->meta.__stride, t1->meta.shape);
                        int coord2 = GET_MIXED_RADIX_DIGIT(i, d, t2->meta.__stride, t2->meta.shape);
                        i1 += (coord1 * t1->meta.stride[d]);
                        i2 += (coord2 * t2->meta.stride[d]);
                    }
                    ((float*)res->data)[i] = ((float*)t1->data)[i1] - ((float*)t2->data)[i2];
                }
        #ifdef OMP
            }
        #endif

}

void _tensor_sub_int32(tensor* t1, tensor* t2, tensor* res){
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
                    int i1 = 0;
                    int i2 = 0;
                    for(uint8 d=0; d < res_meta.dim; d++){
                        int coord1 = GET_MIXED_RADIX_DIGIT(i, d, t1->meta.__stride, t1->meta.shape);
                        int coord2 = GET_MIXED_RADIX_DIGIT(i, d, t2->meta.__stride, t2->meta.shape);
                        i1 += (coord1 * t1->meta.stride[d]);
                        i2 += (coord2 * t2->meta.stride[d]);
                    }
                    ((int*)res->data)[i] = ((int*)t1->data)[i1] - ((int*)t2->data)[i2];
                }
        #ifdef OMP
            }
        #endif

}