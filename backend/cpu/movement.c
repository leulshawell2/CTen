/**
 * this is for ops that do allocate new memory and move data
 */

#include "movement.h"


void tensor_contiguous(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args+1);

    _tensor_contiguous(t, res);

    ERROR_FROM_RES(ctx, res)
}


void tensor_index(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args+1);
    int* idxs = *((int**)(args + 2*sizeof(tensor*)));

    _tensor_index(t, idxs, res);

    ERROR_FROM_RES(ctx, res)
}

void tensor_clone(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args+1);

    _tensor_clone(t, res);
    
    ERROR_FROM_RES(ctx, res)
}
