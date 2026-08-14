#include  "_cpu_impl.h"


void _tensor_mul_float32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        float*d1 = t1->data;
        float*d2 = t2->data;
        float*dr = res->data;

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
                    dr[i] = d1[i1] * d2[i2];
                }
        #ifdef OMP
            }
        #endif

}


void _tensor_div_float32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        float*d1 = t1->data;
        float*d2 = t2->data;
        float*dr = res->data;

        #ifdef OMP
        int nthreads = omp_get_num_threads();
        #pragma omp parallel
        {
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
                    dr[i] = d1[i1] / d2[i2];
                }
        #ifdef OMP
            }
        #endif

}



void _tensor_add_float32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        float*d1 = t1->data;
        float*d2 = t2->data;
        float*dr = res->data;

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
                    dr[i] = d1[i1] + d2[i2];
                }
        #ifdef OMP
            }
        #endif

}

void _tensor_add_int32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        int*d1 = t1->data;
        int*d2 = t2->data;
        int*dr = res->data;

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
                    dr[i] = d1[i1] + d2[i2];
                }
        #ifdef OMP
            }
        #endif

}


void _tensor_sub_float32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        float*d1 = t1->data;
        float*d2 = t2->data;
        float*dr = res->data;


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
                    dr[i] = d1[i1] - d2[i2];
                }
        #ifdef OMP
            }
        #endif

}

void _tensor_sub_int32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        int*d1 = t1->data;
        int*d2 = t2->data;
        int*dr = res->data;


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
                    dr[i] = d1[i1] - d2[i2];
                }
        #ifdef OMP
            }
        #endif

}


void _tensor_matmul_float32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        float*d1 = t1->data;
        float*d2 = t2->data;
        float*dr = res->data;

        

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
                    for(uint8 d=0; d < res_meta.dim-2; d++){
                        int coord = GET_MIXED_RADIX_DIGIT(i, d, res->meta.__stride, res->meta.shape);
                        i1 += (coord * t1->meta.stride[d]);
                        i2 += (coord * t2->meta.stride[d]);
                    }
                    
                    int row = GET_MIXED_RADIX_DIGIT(i, res->meta.dim - 2, res->meta.__stride, res->meta.shape);
                    int col = GET_MIXED_RADIX_DIGIT(i, res->meta.dim - 1, res->meta.__stride, res->meta.shape);
                    

                    i1 += (row * t1->meta.stride[res->meta.dim-2]);
                    i2 += (col * t2->meta.stride[res->meta.dim-1]);
                    
                    int acc = 0;
                    for(int j=0; j < t1->meta.shape[t1->meta.dim - 1]; j++)
                        acc += d1[i1 + j * t1->meta.stride[res->meta.dim-1]] * d2[i2 + j*t2->meta.stride[res->meta.dim-2]];
                    
                    dr[i] = acc;

                }
        #ifdef OMP
            }
        #endif

}



void _tensor_matmul_int32(tensor* t1, tensor* t2, tensor* res){
        tensor_meta res_meta = res->meta;

        int*d1 = t1->data;
        int*d2 = t2->data;
        int*dr = res->data;

        

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
                    for(uint8 d=0; d < res_meta.dim-2; d++){
                        int coord = GET_MIXED_RADIX_DIGIT(i, d, res->meta.__stride, res->meta.shape);
                        i1 += (coord * t1->meta.stride[d]);
                        i2 += (coord * t2->meta.stride[d]);
                    }
                    
                    int row = GET_MIXED_RADIX_DIGIT(i, res->meta.dim - 2, res->meta.__stride, res->meta.shape);
                    int col = GET_MIXED_RADIX_DIGIT(i, res->meta.dim - 1, res->meta.__stride, res->meta.shape);
                    

                    i1 += (row * t1->meta.stride[res->meta.dim-2]);
                    i2 += (col * t2->meta.stride[res->meta.dim-1]);
                    
                    int acc = 0;
                    for(int j=0; j < t1->meta.shape[t1->meta.dim - 1]; j++)
                        acc += d1[i1 + j * t1->meta.stride[res->meta.dim-1]] * d2[i2 + j*t2->meta.stride[res->meta.dim-2]];
                    
                    dr[i] = acc;

                }
        #ifdef OMP
            }
        #endif

}

