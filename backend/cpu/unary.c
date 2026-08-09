/**
 * ops on a single tensor. reduce, dtype
 */
#include "movement.h"

void tensor_cast(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t = EXTRACT_ARG_PTR(tensor*, args + 1);
    int dt = *((int*)(args + 2* sizeof(tensor*)));
    
    tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, dt, NULL, NULL, res);
    
    if(dt == t->meta.dtype){
        _tensor_clone(t, res);
        return;
    }
    switch (dt){
        case Float32:
            _tensor_cast_int32_float32(t, res);
            break;
        case Int32:
            _tensor_cast_float32_int32(t, res);
            break;
    }

    ERROR_FROM_RES(ctx, res)
}



