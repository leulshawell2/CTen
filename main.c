#include "cten.h"


int main(){
    uint dim = 3;
    int shape[5] = {4, 2, 3, 0, 0};

    tensor t = tensor_build(3, shape, sizeof(float), NULL, NULL);
    tensor t2 = tensor_build(3, shape, sizeof(float), &t, NULL);

    tensor_print_meta(t);
    tensor_print_meta(t2);

    tensor_free(t2);
    tensor_print_meta(t);
    // tensor_print_meta(t2);

    tensor_print_data(t);
    // tensor_print_data(t2);

    return 0;
    
}

