#include "libtensor.h"


int main(){
    uint dim = 4;
    int shape[2] = {4, 4};

    tensor t = tensor_build(2, shape, sizeof(float), NULL);

    for(int i=0; i < t.meta.size; i++){
        ((float*)t.data)[i] = i;
    }
    
    tensor_meta m = t.meta;
    // printf("stride=(");
    // for(int i=0; i< dim; i++){
    //     printf("%d, ", m._stride[i]);
    // }
    // printf(") ");
    // printf("shape=(");
    // for(int i=0; i< dim; i++){
    //     printf("%d, ", m.shape[i]);
    // }
    // printf(")\n");
    // for (uint i = 0; i < m.size; i++) {
    //     int i1 = 0;

    //     printf("index=%d, count=( ", i);
    //     for(uint8 d=0; d < m.dim; d++){
    //         int coord = GET_MIXED_RADIX_DIGIT(i, d, m._stride, m.shape);
    //         i1 += (coord * m.stride[d]);
    //         printf("%d, ", coord);
    //     }
    //     printf(")\n");
    // }

    int repeat[2] = {2, 2};
    tensor t2 = tensor_repeat(t, repeat);

    print_meta(t);
    print_meta(t2);
    print_data(t);
    print_data(t2);

    return 0;

}

