#include "view_ops.h"


void tensor_contiguous(context* ctx, op_args args, tensor* res){
    _tensor_contiguous((tensor*)args, res);
}


void tensor_index(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    int* idxs = *((int**)(args + sizeof(tensor*)));
    _tensor_index(t, idxs, res);
}


void tensor_transpose(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    int* idxs = *((int**)(args + sizeof(tensor*)));
    _tensor_transpose(t, idxs[0], idxs[1], res);
}



void tensor_permute(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    uint8* idxs = *((uint8**)(args + sizeof(tensor*)));
    _tensor_permute(t, idxs, res);
}


void tensor_view(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    args = args + sizeof(tensor);
    int dim = *EXTRACT_ARG_PTR(int, args);
    int* dims = *((int**)(args + sizeof(int*)));
    _tensor_view(t, dims, dim, res);
}

