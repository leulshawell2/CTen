#include "ops.c"

int main(){
    uint dim = 3;
    int shape[dim];

    for(uint8 d =0; d < dim; d++){
        shape[d] = 3;
    }

    tensor t = build_tensor(dim, shape, sizeof(float), NULL);
    
    tensor t2 = build_tensor(dim, shape, sizeof(float), NULL);
    tensor t4 = build_tensor(dim, shape, sizeof(float), t2.data);

    tensor_transpose(t2, 0, 1);

    tensor tr = tensor_contiguous(t2);

    tensor t3 = build_tensor(dim, shape, sizeof(float), NULL);

    add_float(t, t2, t3);

    print_meta(t);
    print_data(t);
    print_meta(t2);
    print_data(t2);
    print_meta(tr);
    print_data(tr);
    print_meta(t4);
    print_data(t4);
    print_meta(t3);
    print_data(t3);

}

