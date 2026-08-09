#ifdef OMP
#include <omp.h>
#endif

#include  "_cpu_impl.h"


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
    tensor_build(dim, shape, t->meta.e_size, None, t, NULL, res);
}

void _tensor_reshape(tensor* t, int* shape, int dim, tensor* res){
    if(tensor_iscontiguous(t))
        _tensor_view(t, shape, dim, res);

    tensor temp;
    _tensor_contiguous(t, &temp);
    _tensor_view(&temp, shape, dim, res);

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

    tensor_build(dim, new_shape, t->meta.e_size, None, t, NULL, res);
    memcpy(res->meta.stride, new_stride, dim * sizeof(int));

}


void _tensor_broadcast_match(tensor* _t1, tensor* _t2){
    int max_dim = MAX(_t1->meta.dim, _t2->meta.dim);
    int new_shape[max_dim];

    if(_t1->meta.dim == _t2->meta.dim){
        for(int d=0; d < max_dim; d++)
            new_shape[d] = _t1->meta.shape[d] == 1? _t2->meta.shape[d]: _t1->meta.shape[d];
    }else {
        int dim_diff;
        int* max_shape;
        int* min_shape;
        if (max_dim == _t1->meta.dim){
            max_shape = _t1->meta.shape;
            min_shape = _t2->meta.shape;
            dim_diff = _t1->meta.dim - _t2->meta.dim;
        }else {
            max_shape = _t2->meta.shape;
            min_shape = _t1->meta.shape;
            dim_diff = _t2->meta.dim - _t1->meta.dim;
        }


        for(int d=0; d < max_dim; d++){
            int s1 = max_shape[d];
            new_shape[d] = d < dim_diff? s1: s1 == 1? min_shape[d-dim_diff]: s1;
            printf("%d ", new_shape[d]);
        };
    }
}


void _tensor_transpose(tensor* t, uint8 _dim1, uint8 _dim2, tensor* res){
    tensor_meta m = t->meta;
    uint8 dim1 = m.dim - _dim1 - 1;
    uint8 dim2 = m.dim - _dim2 - 1;
    
    tensor_build(t->meta.dim, t->meta.shape, None, t->meta.e_size, t, NULL, res);
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

    tensor_build(m.dim, m.shape, m.e_size, None, t, NULL, res);
    memcpy(res->meta.stride, m.stride, sizeof(int) * m.dim);

    memcpy(new_shape, m.shape, m.dim * sizeof(int));
    for(uint8 d=0; d < m.dim; d++){
        d2 = m.dim - dims[d] -1;
        res->meta.shape[d] = m.shape[d2];
        res->meta.stride[d] = m.stride[d2];
    }
    
    meta_stride(res->meta.shape, res->meta.__stride, res->meta.dim);
}

