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
    
    
    int shape[4] = {1, 3, 4, 3};
    tensor t ;
    tensor t1;
    tensor t2;
    tensor t3;

    tensor_build(3, shape, sizeof(float), Int32, NULL, NULL, &t);
    
    
    shape[1] = 1;
    shape[2] = 1;
    shape[3] = 1;
    
    tensor_build(4, shape, sizeof(float), Int32, NULL, NULL, &t1);
    tensor_print_meta(&t);
    tensor_print_meta(&t1);
    _tensor_broadcast_match(&t, &t1, &t, &t2);
    
    tensor_print_meta(&t);
    tensor_print_meta(&t1);
    
    return 0;

}



