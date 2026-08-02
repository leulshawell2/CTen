#include "libtensor.h"


int main(){
    uint dim = 4;
    int shape[dim];

    shape[0] = 1;
    shape[1] = 256;

    tensor t = tensor_build(2, shape, sizeof(float), NULL);


    for(uint8 d =0; d < dim; d++){
        shape[d] = 4;
    }

    t.meta.stride[0] = 0;


    print_meta(t);

    tensor t2 = tensor_view(t, shape, 4);

    print_meta(t2);

    return 0;

}

