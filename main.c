
#include "cten.h"

#define OMP
#include <omp.h>


tensor contiguous_handler(context* ctx, op_args args, tensor* res){
    tensor_contiguous((tensor*)args, res);
}


tensor index_handler(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    int* idxs = *((int**)(args + sizeof(tensor*)));
    tensor_index(t, idxs, res);
}


tensor trans_handler(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    int* idxs = *((int**)(args + sizeof(tensor*)));
    tensor_transpose(t, idxs[0], idxs[1], res);
}



tensor permute_handler(context* ctx, op_args args, tensor* res){
    tensor* t = EXTRACT_ARG_PTR(tensor, args);
    uint8* idxs = *((uint8**)(args + sizeof(tensor*)));
    tensor_permute(t, idxs, res);
}






int main(){

    int nthreads = omp_get_num_threads();

    printf("OMP using %d threads\n", nthreads);

    context* cpu_ctx = CT_context_init(2);
    CT_register_op(cpu_ctx, OP_CONT, contiguous_handler);
    CT_register_op(cpu_ctx, OP_INDEX, index_handler);
    
    
    int shape[3] = {3, 4, 2};
    tensor t ;
    tensor t1;
    tensor t2;

    tensor_build(3, shape, sizeof(float), NULL, NULL, &t);

    
    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }
    int repeat[3] = {2, 2, 0};

    tensor_repeat(&t, repeat, &t1);
    tensor_print_data(&t);
    tensor_print_data(&t1);

    return 0;

}



