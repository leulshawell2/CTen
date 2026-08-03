#include "libtensor.h"


int main(){
    uint dim = 3;
    int shape[5] = {4, 2, 3, 0, 0};

    tensor t = tensor_build(3, shape, sizeof(float), NULL);
        
    
    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }
    
    int repeat[3] = {2, 2, 2};
    tensor t2 = tensor_repeat(t, repeat);
    
    shape[0] = 1;
    shape[1] = 192;
    shape[2] = 1;
    tensor t3 = tensor_view(t2, shape, 3);

    shape[0] = 2;
    shape[1] = 192;
    shape[2] = 1;
    shape[3] = 4;
    shape[4] = 4;

    tensor t4 = tensor_broadcast(t3, shape, 3);
    
    tensor_print_meta(t);
    tensor_print_meta(t2);
    tensor_print_meta(t3);
    tensor_print_meta(t4);

    tensor_print_data(t4);

    return 0;

}

