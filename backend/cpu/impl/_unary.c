#include "_cpu_impl.h"
#include <math.h>
#include <omp.h>

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



void _tensor_max_f32(tensor* t, tensor* res){
    tensor_meta res_meta = res->meta;

    float* dr = res->data;
    float* d = t->data;

    int dim = t->meta.dim - 1;

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
            int i1 = 0;
            for(uint8 d=0; d < dim; d++){
                int coord = GET_MIXED_RADIX_DIGIT(i, d, res->meta.__stride, res->meta.shape);
                i1 += (coord * t->meta.stride[d]);
            }

            dr[i]= -__builtin_inff32();
            for(int j=0; j < t->meta.shape[dim]; j++){
                int cand = d[i1 + j * t->meta.stride[dim]];
                dr[i] = MAX(d[i], cand);
            }
        }
    }


}


void _tensor_min_f32(tensor* t,  tensor* res){
    tensor_meta res_meta = res->meta;

    float* dr = res->data;
    float* d = t->data;

    int dim = t->meta.dim - 1;

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
            int i1 = 0;
            for(uint8 d=0; d < dim; d++){
                int coord = GET_MIXED_RADIX_DIGIT(i, d, res->meta.__stride, res->meta.shape);
                i1 += (coord * t->meta.stride[d]);
            }

            dr[i]= __builtin_inff32();
            for(int j=0; j < t->meta.shape[dim]; j++){
                int cand = d[i1 + j * t->meta.stride[dim]];
                dr[i] = MIN(d[i], cand);
            }
        }
    }


}

void _tensor_map(tensor* t,  mapper map, tensor* res){
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
                    map(((char*)t->data) + i * t->meta.e_size, ((char*)res->data) + i * t->meta.e_size);
                }
        #ifdef OMP
            }
        #endif

}
