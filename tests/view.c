#include "../backend/cpu/cpu.h"



void view_f32(){

    tensor _t;
    tensor _t1;
    tensor _t2;

    tensor* t =  &_t;
    tensor* t1 = &_t1;

    int shape[3] = {3, 2, 4};
    
    tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, t);
    for(int i=0; i < t->meta.size; i++){
        ((float*)t->data)[i] = i;
    }
    // tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, t1);
    
    //===========================================
    int new_dims = 2;
    int new_shape[2] = {2, 12};
    _tensor_transpose(t, 1,  0, t1);
    //===========================================


    //===============================================
    // tensor_build(3, shape, sizeof(float), Float32, NULL, NULL, t1);
    int idxs[9] = {1, 3, 1,   0, 2, 1,   0, 4, 1};
    // _tensor_index(t, idxs, t1);
    //===============================================

    tensor_print_meta(t);
    tensor_print_meta(t1);

    tensor_print_data(t);
    tensor_print_data(t1);

}