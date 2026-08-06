#include "view_ops.h"


void tensor_contiguous(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args + sizeof(tensor*));
    _tensor_contiguous(t, res);
}


void tensor_index(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args+1);
    int* idxs = *((int**)(args + sizeof(tensor*)));
    _tensor_index(t, idxs, res);
}


void tensor_transpose(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t   = EXTRACT_ARG_PTR(tensor*, args+1);
    int* dims      = (int*)(args + 2*sizeof(tensor*));
    _tensor_transpose(t, dims[0], dims[1], res);
}



void tensor_permute(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t   = EXTRACT_ARG_PTR(tensor*, args+1);
    uint8* idxs = (uint8*)(args + 2*sizeof(tensor*));
    _tensor_permute(t, idxs, res);
}


void tensor_view(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t   = EXTRACT_ARG_PTR(tensor*, args+1);
    int  dim    = *((int*)(args + 2*sizeof(tensor*)));
    int* dims   =  (int*)(args + 2*sizeof(tensor*) + sizeof(int));
    _tensor_view(t, dims, dim, res);
}

