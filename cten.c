#include "cten.h"



#ifdef OMP
#include <omp.h>
#endif



tensor tensor_contiguous(tensor t){

    if (tensor_iscontiguous(t)){
        return t;
    }
    
    tensor t1 = tensor_build(t.meta.dim, t.meta.shape, t.meta.e_size, NULL, NULL);
    tensor_meta m1 = t1.meta;

    #ifdef OMP
        #pragma omp parallel
        {
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int w_ = (m1.size + nthreads-1) / nthreads;
            int start  = tid * w_;
            int end = start + w_;
            if (end > m1.size){
                end = m1.size;
            }
            for (int i = start; i < end; i++) {

    #else
            for (int i = 0; i < m1.size; i++) {
    #endif
                int i1 = 0;
                for(uint8 d=0; d < m1.dim; d++){
                    int coord = GET_MIXED_RADIX_DIGIT(i, d, m1.__stride, m1.shape);
                    i1 += (coord * t.meta.stride[d]);
                }
                void* des_pos = t1.data + m1.e_size * i;
                void* src_pos = t.data + m1.e_size * i1;
                memcpy(des_pos, src_pos, m1.e_size);

                // ((float*)t1.data)[i] = ((float*)t.data)[i1];
            }
    #ifdef OMP
        }
#endif

    return t1;
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

