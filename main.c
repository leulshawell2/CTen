#include "libtensor.h"


int main(){
    uint dim = 3;
    int shape[3] = {4, 2, 3};

    tensor t = tensor_build(3, shape, sizeof(float), NULL);
        
    
    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }
    
    int repeat[3] = {2, 2, 2};
    tensor t2 = tensor_repeat(t, repeat);
    tensor_print_meta(t);
    tensor_print_meta(t2);
    
    tensor_print_data(t);
    tensor_print_data(t2);
    
    return 0;

    shape[0] = 1;
    shape[1] = 64;
    tensor t3 = tensor_view(t2, shape, 2);
    
    tensor_print_meta(t);
    tensor_print_meta(t2);
    tensor_print_meta(t3);
    tensor_print_data(t2);
    tensor_print_data(t3);

    return 0;

}

