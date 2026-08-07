#include  "_cpu_impl.h"


void _tensor_contiguous(tensor* t, tensor* res){

    if (tensor_iscontiguous(t)){
        tensor_meta_clone(res, t);
        *(t->meta.ref) += 1;
        res->data = t->data;
        return;
        
    }else{
        tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t->meta.dtype, NULL, NULL, res);
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
                    for(uint8 d=0; d < res_meta.dim; d++){
                        int coord = GET_MIXED_RADIX_DIGIT(i, d, res_meta.__stride, res_meta.shape);
                        i1 += (coord * t->meta.stride[d]);
                    }
                    void* des_pos = res->data + res_meta.e_size * i;
                    void* src_pos = t->data + res_meta.e_size * i1;
                    memcpy(des_pos, src_pos, res_meta.e_size);
                }
        #ifdef OMP
            }
        #endif
    }

}

void _tensor_clone(tensor* t, tensor* res){
    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t->meta.dtype, NULL, NULL, res);
    memcpy(res->meta.stride, t->meta.stride, t->meta.dim * sizeof(int));
    memcpy(res->data, t->data, res->meta.e_size * res->meta.size);
}



void _tensor_copy(tensor* t, tensor* res){
    if(tensor_iscontiguous(t))
        _tensor_clone(t, res);
    else
        _tensor_contiguous(t, res);

}

void _tensor_repeat(tensor* t, int* repeat, tensor* res){
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
    tensor_build(temp_dim, shape, t->meta.e_size, None, t, NULL, temps);
    temps->meta.stride = strides;

    tensor_build(temp_dim, shape, t->meta.e_size, None, NULL, NULL, temps + 1);
    _tensor_contiguous(temps, temps + 1);
    tensor_build(t->meta.dim, new_shape, t->meta.e_size, None, temps + 1, NULL, res);

    tensor_free(temps);
    tensor_free(temps + 1);
}   

void _tensor_index(tensor* t, int* idxs, tensor* res){
    int new_shape[t->meta.dim];
    int offset = 0;
    int new_strides[t->meta.dim];

    for(int d=t->meta.dim * 3 - 1; d > 0;  d-=3 ){
        int s = idxs[d-2];
        int e = idxs[d-1];
        int j = idxs[d];

        int d_3 = d/3;
        
        new_shape[d_3] = MAX(0, (e - s - 1)/ j + 1);
        new_strides[d_3] = j * t->meta.stride[d_3];

        offset += t->meta.stride[d_3] * s;
    }
    tensor temp_;
    tensor* temp = &temp_;
    tensor_build(t->meta.dim, new_shape, t->meta.e_size, None, t, NULL, temp);

    temp->meta.stride = new_strides;
    temp->data += offset * temp->meta.e_size;

    _tensor_contiguous(temp, res);
    tensor_free(temp);
}

