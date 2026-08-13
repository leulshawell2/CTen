
#include "cten.h"



context* CT_context_init(int ops_count){

    size_t ctx_size = sizeof(context) + sizeof(op_handler) * ops_count;

    context* ctx = malloc(ctx_size);
    if(ctx == NULL){
        CT_ERROR(MEM_ERR, MALLOC_ERR, ctx);
        return ctx;
    }
    memset(ctx, 0, ctx_size);
    return ctx;
}

void  CT_register_op(context* ctx, int opcode, op_handler handler){
    if(ctx->ops_table[opcode] != NULL){
        CT_ERROR(OP_REG_ERR, FOUND_ERR, ctx);
        return;
    }
    ctx->ops_table[opcode] = handler;
}


void  CT_replace_op(context* ctx, int opcode, op_handler handler){
    if(ctx->ops_table + opcode == NULL){
        CT_ERROR(OP_REG_ERR, NOTFOUND_ERR, ctx);
        return;
    }
    ctx->ops_table[opcode] = handler;
}


op* CT_op_build(context* ctx,  size_t args_size, int opcode, op* next, op* prev){
    op* op = malloc(sizeof(op));
    op->args = malloc(args_size);
    op->opcode = opcode;
    op->next = (struct op*)next;
    op->prev = (struct op*)prev;
    return op;
}


void CT_dispatch_graph(context* ctx, op* entry){
    op* current = entry;

    do{
        if(current->prev == NULL || ((op*)current->prev)->completed){
            op_handler handler =  ctx->ops_table[current->opcode];
            handler(ctx, current->args);

            current = (op*)(current->next);
        }else {
            current = (op*)(current->prev);
        }
    }while(current != NULL);

}
