#include "binary.h"

void tensor_add(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t1 = EXTRACT_ARG_PTR(tensor*, args+1);
    tensor* t2 = EXTRACT_ARG_PTR(tensor*, args+2);

    tensor_build(t1->meta.dim, t1->meta.shape, t1->meta.e_size, t1->meta.dtype, NULL, NULL, res);


    switch (t1->meta.dtype)
    {
        case Float32:
            _tensor_add_float32(t1, t2, res);
            break;
        case Int32:
            _tensor_add_int32(t1, t2, res);
            break;
        default:
            CT_ERROR(DTYPE_ERROR, NOTFOUND_ERR, ctx);
            return;
    }
    ERROR_FROM_RES(ctx, res)
}


void tensor_sub(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t1 = EXTRACT_ARG_PTR(tensor*, args+1);
    tensor* t2 = EXTRACT_ARG_PTR(tensor*, args+2);

    tensor_build(t1->meta.dim, t1->meta.shape, t1->meta.e_size, None, NULL, NULL, res);

    switch (res->meta.dtype)
    {
        case Float32:
            _tensor_sub_float32(t1, t2, res);
            break;
        case Int32:
            _tensor_sub_int32(t1, t2, res);
            break;
        default:
            CT_ERROR(DTYPE_ERROR, NOTFOUND_ERR, ctx);
            return;
    }
    ERROR_FROM_RES(ctx, res)
}


void tensor_matmul(context* ctx, op_args args){
    tensor* res = EXTRACT_ARG_PTR(tensor*, args);
    tensor* t1 = EXTRACT_ARG_PTR(tensor*, args+1);
    tensor* t2 = EXTRACT_ARG_PTR(tensor*, args+2);

    tensor_build(t1->meta.dim, t1->meta.shape, t1->meta.e_size, None, NULL, NULL, res);

    switch (res->meta.dtype)
    {
        case Float32:
            _tensor_matmul_float32(t1, t2, res);
            break;
        case Int32:
            _tensor_matmul_int32(t1, t2, res);
            break;
        default:
            CT_ERROR(DTYPE_ERROR, NOTFOUND_ERR, ctx);
            return;
    }
    ERROR_FROM_RES(ctx, res)
}
