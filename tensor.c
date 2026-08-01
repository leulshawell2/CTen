#include <stdlib.h>
#include <string.h>

#include  "meta.c"


int size(tensor t){
    int size = 1;
    for(uint8 d=0; d < t.meta.dim; d++) 
        size = size * t.meta.shape[d];
    return size;
}

void print_data(tensor t){
    tensor_meta m = t.meta;
    
    for (uint i = 0; i < m.size; i++) {
        int i1 = 0;

        for(uint8 d=0; d < m.dim; d++){
            int coord = GET_MIXED_RADIX_DIGIT(i, d, m._stride, m.shape);
            i1 += (coord * m.stride[d]);
        }
        printf("%f,\t", ((float*)t.data)[i1]);
    }
    printf("\n");
}


tensor build_tensor(uint8 dim, int* shape, uint8 e_size, void* data){
    tensor t;
    t.meta.dim = dim;
    t.meta.e_size = e_size;

    int size = 1;
    for(uint8 d=0; d < dim; d++) 
        size = size * shape[d];
    t.meta.size = size;


    int meta_size = sizeof(int) * dim;
    
    void* meta = malloc(meta_size * 3);

    t.meta.shape = meta;
    memcpy(t.meta.shape, shape, meta_size);
    
    t.meta.stride = meta + meta_size;
    t.meta._stride = meta + 2 * meta_size;

    t.meta.stride[0] = 1;
    for (int8 d = 1; d < dim; d++)
        t.meta.stride[d] = shape[d] * t.meta.stride[d-1];

    memcpy(t.meta._stride, t.meta.stride, meta_size);
    
    if (!data){
        t.data = malloc(size * e_size);
    }

    for(int i=0; i < size; i++){
        ((float*)t.data)[i] = i;
    }

    return t;
}


void free_tensor(tensor t) {
    free(t.meta.shape);
}




