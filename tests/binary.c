#include "../backend/cpu/cpu.h"



void binary_f32(){

tensor _t;
    tensor _t1;
    tensor _t2;

    tensor* t = &_t;
    tensor* t1 = &_t1;
    tensor* t2 = &_t2;


    int shape[3] = {4, 3, 3};

    tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, t);
    tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, t1);
    tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, t2);


    for(int i=0; i < t->meta.size; i++){
        ((float*)t->data)[i] = i;
        ((float*)t1->data)[i] = i;
    }

    _tensor_add_float32(t, t1,  t2);
    _tensor_sub_float32(t, t1,  t2);
    _tensor_matmul_float32(t, t1,  t2);

    tensor_print_meta(t);
    tensor_print_meta(t2);

    tensor_print_data(t);
    tensor_print_data(t1);
    tensor_print_data(t2);

}