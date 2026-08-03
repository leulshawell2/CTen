#include "cten.h"


int main(){
    uint dim = 3;
    int shape[3] = {3, 2, 4};


    //create a new tensor
    tensor t = tensor_build(3, shape, sizeof(float), NULL, NULL);

    //fill the tensor for testing
    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }

    
    shape[0] = 2;
    shape[1] = 3;
    shape[2] = 1;
    // tensor t2 = tensor_view(t, shape, 2);
    uint8 dims[3] = {1, 0, 2};
    tensor t2 = tensor_permute(t, dims);


    tensor_print_meta(t);
    tensor_print_data(t);

    tensor_print_meta(t2);
    tensor_print_data(t2);


    
    return 0;
    
}



