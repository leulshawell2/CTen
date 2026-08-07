
#include "cten.h"



context* CT_context_init(int ops_count){

    size_t ctx_size = sizeof(context) + sizeof(op_handler) * ops_count;

    context* ctx = malloc(ctx_size);
    if(ctx == NULL){
        CT_ERROR(MEM_ERR, MALLOC_ERR, ctx);
    }
    memset(ctx, 0, ctx_size);
    return ctx;
}

void  CT_register_op(context* ctx, int op_id, op_handler handler){
    if(ctx->ops_table[op_id] != NULL){
        CT_ERROR(OP_REG_ERR, FOUND_ERR, ctx);
        return;
    }
    ctx->ops_table[op_id] = handler;
}


void  CT_replace_op(context* ctx, int op_id, op_handler handler){
    if(ctx->ops_table + op_id == NULL){
        CT_ERROR(OP_REG_ERR, NOTFOUND_ERR, ctx);
        return;
    }
    ctx->ops_table[op_id] = handler;
}

