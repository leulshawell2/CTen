#define OMP


#include "./cpu/view_ops.h"



int main(){

    context* cpu_ctx = CT_context_init(2);
    CT_register_op(cpu_ctx, OP_CONT, tensor_contiguous);
    CT_register_op(cpu_ctx, OP_INDEX, tensor_index);
    
    
    int shape[3] = {3, 4, 2};
    tensor t ;
    tensor t1;
    tensor t2;

    tensor_build(3, shape, sizeof(float), NULL, NULL, &t);

    
    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }
    int repeat[4] = {3, 2, 2, 2};

    _tensor_view(&t, repeat, 4, &t1);
    tensor_print_meta(&t);
    tensor_print_meta(&t1);
    tensor_print_data(&t);
    printf("\n\n");
    tensor_print_data(&t1);

    return 0;

}



