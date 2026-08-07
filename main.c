#define OMP


#include "./backend/cpu/cpu.h"



int main(){

    #ifdef OMP
        int n_threads = omp_get_num_threads();
        printf("using %d OMP threads\n", n_threads);
    #endif

    context* cpu_ctx = CT_context_init(2);
    CT_register_op(cpu_ctx, OP_CONTIGUOUS,  tensor_contiguous);
    CT_register_op(cpu_ctx, OP_INDEX, tensor_index);
    
    
    int shape[3] = {3, 4, 3};
    tensor t ;
    tensor t1;
    tensor t2;
    tensor t3;

    tensor_build(3, shape, sizeof(float), Int32, NULL, NULL, &t);

    
    for(int i=0; i < t.meta.size; i++){
        ((int*)t.data)[i] = i;
    }
    
    op_args args = malloc(3 * sizeof(tensor*));

    ((tensor**)args)[0] = &t2;
    ((tensor**)args)[1] = &t;
    ((tensor**)args)[2] = &t;
    
    tensor_add(cpu_ctx, args);
    tensor_print_data(&t);
    tensor_print_data(&t2);

    return 0;

}



