#define OMP

#include <stdio.h>
#include "./backend/cpu/cpu.h"





int main(){

    core_context ctx;

    init_cpu_context(&ctx);
    

    tensor t;

    t.meta.ctx = &ctx;

    int shape[3] = {1, 1, 1};

    tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, &t);

    printf("%p %p", t.data, t.meta.shape);


    tensor_print_data(&t);
    return 0;

}






