#include "./impl/_view_ops.h"


void tensor_contiguous(context* ctx, op_args args, tensor* res);
void tensor_index(context* ctx, op_args args, tensor* res);
void tensor_transpose(context* ctx, op_args args, tensor* res);
void tensor_permute(context* ctx, op_args args, tensor* res);