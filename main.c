#include "ctensor.h"


int main(){
    uint dim = 3;
    int shape[dim];

    for(uint8 d =0; d < dim; d++){
        shape[d] = 3;
    }

    tensor t = build_tensor(dim, shape, sizeof(float), NULL);
    
    tensor clone = tensor_clone(t); 
    tensor copy = tensor_copy(t); 


    print_data(t);
    print_data(clone);
    print_data(copy);
    return 0;

}

