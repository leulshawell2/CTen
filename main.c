#include "ctensor.h"


int main(){
    uint dim = 2;
    int shape[dim];

    for(uint8 d =0; d < dim; d++){
        shape[d] = 3;
    }

    tensor t = build_tensor(dim, shape, sizeof(float), NULL);
    tensor tt = build_tensor(dim, shape, sizeof(float), NULL);
    tensor t2 = build_tensor(dim, shape, sizeof(float), NULL);



    print_data(t2);

    return 0;

}

