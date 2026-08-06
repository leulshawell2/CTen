#define OMP


#include "./backend/cpu/cpu.h"



int main(){

    context* cpu_ctx = CT_context_init(2);
    CT_register_op(cpu_ctx, OP_CONTIGUOUS,  tensor_contiguous);
    CT_register_op(cpu_ctx, OP_INDEX, tensor_index);
    
    
    int shape[3] = {3, 4, 3};
    tensor t ;
    tensor t1;
    tensor t2;

    tensor_build(3, shape, sizeof(float), NULL, NULL, &t);

    
    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }
    
    op_args args = malloc(2 * sizeof(tensor*) + 2 * sizeof(int));

    ((tensor**)args)[0] = &t2;
    ((tensor**)args)[1] = &t;
    ((int*)(args + 2*sizeof(tensor*)))[0] = 0;
    ((int*)(args + 2*sizeof(tensor*)))[1] = 1;

    uint8 permute[3] = {0, 2, 1};
    _tensor_permute(&t, permute, &t2);
    _tensor_contiguous(&t2, &t1);

    tensor_print_data(&t);
    tensor_print_data(&t2);

    return 0;

}



