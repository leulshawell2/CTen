#include "cten.h"


int main(){
    int shape[2] = {3, 3};

    tensor t = tensor_build(2, shape, sizeof(float), NULL, NULL);

    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }



    int idxs[4] = {0, 1,  1, 2};

    tensor t2 = tensor_index(t, idxs);

    // idxs[0] = 0;
    // idxs[1] = 1;   
    // idxs[2] = 1;   
    // idxs[3] = 2;   

    // tensor t3 = tensor_index(t, idxs);


    tensor_print_meta(t);
    tensor_print_meta(t2);
    
    tensor_print_data(t);
    tensor_print_data(t2);
    // tensor_print_data(t3);
    
    return 0;
    
}



