#ifdef OMP
#include <omp.h>
#endif

#include  "_view_ops.h"



void _tensor_contiguous(tensor* t, tensor* res){

    if (tensor_iscontiguous(t)){
        tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t, NULL, res);
        return;
        
    }else{
        tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, NULL, NULL, res);
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
    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, NULL, NULL, res);
    memcpy(res->meta.stride, t->meta.stride, t->meta.dim * sizeof(int));
    memcpy(res->data, t->data, res->meta.e_size * res->meta.size);
}



void _tensor_copy(tensor* t, tensor* res){
    _tensor_contiguous(t, res);

}

void _tensor_view(tensor* t, int* shape, int dim, tensor* res){

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

void _tensor_reshape(tensor* t, int* shape, int dim, tensor* res){
    if(tensor_iscontiguous(t))
        _tensor_view(t, shape, dim, res);

    tensor temp;
    _tensor_contiguous(t, &temp);
    _tensor_view(&temp, shape, dim, res);

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
    tensor_build(temp_dim, shape, t->meta.e_size, t, NULL, temps);
    temps->meta.stride = strides;

    tensor_build(temp_dim, shape, t->meta.e_size, NULL, NULL, temps + 1);
    _tensor_contiguous(temps, temps + 1);
    tensor_build(t->meta.dim, new_shape, t->meta.e_size, temps + 1, NULL, res);

    tensor_free(temps);
    tensor_free(temps + 1);
}   

void _tensor_broadcast(tensor* t, int* shape, int dim, tensor* res){
    int new_shape[dim];
    int new_stride[dim];

    int new_dims = dim - t->meta.dim;

    for(int d=0; d < dim; d++ ){
        if(d < new_dims){
            new_shape[d] = shape[d];
            new_stride[d] = 0;
        }else {
            int d2 = d - new_dims;
            new_shape[d] = shape[d] == 1? t->meta.shape[d2]: shape[d];
            new_stride[d] = shape[d] == 1? t->meta.stride[d2]: 0;    
        }
    }

    tensor_build(dim, new_shape, t->meta.e_size, t, NULL, res);
    memcpy(res->meta.stride, new_stride, dim * sizeof(int));
    

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
    tensor_build(t->meta.dim, new_shape, t->meta.e_size, t, NULL, temp);

    temp->meta.stride = new_strides;
    temp->data += offset * temp->meta.e_size;

    _tensor_contiguous(temp, res);
    tensor_free(temp);
}



void _tensor_transpose(tensor* t, uint8 _dim1, uint8 _dim2, tensor* res){
    tensor_meta m = t->meta;
    uint8 dim1 = m.dim - _dim1 - 1;
    uint8 dim2 = m.dim - _dim2 - 1;
    
    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t, NULL, res);
    memcpy(res->meta.stride, t->meta.stride,  m.dim * sizeof(int));
    
    
    
    int tmp = res->meta.shape[dim1];
    res->meta.shape[dim1] = res->meta.shape[dim2];
    res->meta.shape[dim2] = tmp;
    
    meta_stride(res->meta.shape, res->meta.__stride, res->meta.dim);
    
    tmp = m.stride[dim1];
    res->meta.stride[dim1] = m.stride[dim2];
    res->meta.stride[dim2] = tmp;



}


void _tensor_permute(tensor* t, uint8* dims, tensor* res){
    tensor_meta m = t->meta;

    int new_shape[m.dim];
    int new_strides[m.dim];
    int d2;

    tensor_build(m.dim, m.shape, m.e_size, t, NULL, res);
    memcpy(res->meta.stride, m.stride, sizeof(int) * m.dim);

    memcpy(new_shape, m.shape, m.dim * sizeof(int));
    for(uint8 d=0; d < m.dim; d++){
        d2 = m.dim - dims[d] -1;
        res->meta.shape[d] = m.shape[d2];
        res->meta.stride[d] = m.stride[d2];
    }
    
    meta_stride(res->meta.shape, res->meta.__stride, res->meta.dim);
}

