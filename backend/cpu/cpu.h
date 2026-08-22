#include "./impl/_cpu_impl.h"


#define OPS_COUT 7

void init_cpu_context(core_context* ctx){
    ctx->data_alloc = malloc;
    ctx->meta_alloc = malloc;
    ctx->data_free =  free;
    ctx->meta_free =  free;
    ctx->data_copy =  memcpy;
    ctx->meta_copy =  memcpy;
    ctx->data_cmp =   memcmp;
    ctx->meta_cmp =   memcmp;
}






void tensor_contiguous(context* ctx, op_args args);
void tensor_index(context* ctx, op_args args);
void tensor_transpose(context* ctx, op_args args);
void tensor_permute(context* ctx, op_args args);
void tensor_add(context* ctx, op_args args);
void tensor_sub(context* ctx, op_args args);
void tensor_cast(context* ctx, op_args args);