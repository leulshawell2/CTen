#include "../backend/cpu/cpu.h"



void binary_f32(){

    tensor _t;
    tensor _t1;
    tensor _t2;
    tensor _t3;

    tensor* t = &_t;
    tensor* t1 = &_t1;
    tensor* t2 = &_t2;
    tensor* t3 = &_t3;


    int shape[4] = {2, 2, 3, 3};
    
    tensor_build(4, shape, sizeof(float), Float32, NULL, NULL, t);
    shape[3] = 1;
    tensor_build(4, shape, sizeof(float), Float32, NULL, NULL, t1);
    tensor_build(4, shape, sizeof(float), Float32, NULL, NULL, t3);


    for(int i=0; i < t->meta.size; i++){
        ((float*)t->data)[i]  = i;
        // ((float*)t1->data)[i] = i;
    }
    
    
    _tensor_transpose(t, 0, 2, t2);
    tensor_print_data(t);
    tensor_print_data(t2);
    return;
    _tensor_max_f32(t2, t3);
    _tensor_transpose(t3, 0, 1, t1);

    tensor_print_data(t1);


}