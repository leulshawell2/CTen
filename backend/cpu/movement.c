/**
 * this is for ops that do allocate nre memory and move data
 */

#include "movement.h"


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

void tensor_clone(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args + sizeof(tensor*));
    _tensor_clone(t, res);
}
