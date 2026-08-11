#include "./impl/_cpu_impl.h"


#define OPS_COUT 7

void tensor_contiguous(context* ctx, op_args args);
void tensor_index(context* ctx, op_args args);
void tensor_transpose(context* ctx, op_args args);
void tensor_permute(context* ctx, op_args args);
void tensor_add(context* ctx, op_args args);
void tensor_sub(context* ctx, op_args args);
void tensor_cast(context* ctx, op_args args);