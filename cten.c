
#include "cten.h"



#ifdef OMP
#include <omp.h>
#endif



void tensor_contiguous(tensor* t, tensor* res){

    if (tensor_iscontiguous(t)){
        tensor_build(t->meta.dim, t->meta.shape, t->meta.e_size, t, NULL, res);
        return;

    }else{
        tensor_meta res_meta = res->meta;

        #ifdef OMP
            #pragma omp parallel
            {
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();

                int w_ =  res_meta.size + nthreads-1) / nthreads;
                int start  = tid * w_;
                int end = start + w_;
                if (end > res_meta.size){
                    end = res_meta.size;
                }
                for (int i = start; i < end; i++) {

        #else
                for (int i = 0; i < res_meta.size; i++) {
        #endif
                    int i1 = 0;
                    for(uint8 d=0; d < res_meta.dim; d++){
                        int coord = GET_MIXED_RADIX_DIGIT(i, d, res_meta.__stride, res_meta.shape);
                        i1 += (coord * t->meta.stride[d]);
                    }
                    void* des_pos = res->data + res_meta.e_size * i;
                    void* src_pos = t->data + res_meta.e_size * i1;
                    memcpy(des_pos, src_pos, res_meta.e_size);
                }
        #ifdef OMP
            }
        #endif
    }

}



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

void  CT_register_op(context* ctx, int op_id, op_handler handler){
    if(ctx->ops_table[op_id] != NULL){
        CT_ERROR(OP_REG_ERR, OP_FOUND_ERR, ctx);
    }

    ctx->ops_table[op_id] = handler;
}


void  CT_replace_op(context* ctx, int op_id, op_handler handler){
    if(ctx->ops_table + op_id == NULL){
        CT_ERROR(OP_REG_ERR, OP_NOTFOUND_ERR, ctx);
    }
    
    ctx->ops_table[op_id] = handler;
}

