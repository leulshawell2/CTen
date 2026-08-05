#include "cten.h"





tensor contiguous_handler(context* ctx, op_args args){
    return tensor_contiguous(*(tensor*)args);
}


tensor index_handler(context* ctx, op_args args){
    tensor t = (*(tensor**)args)[0];
    int* idxs = *((int**)(args + sizeof(tensor*)));
    return tensor_index(t, idxs);
}





int main(){
    

    context* ctx = CT_context_init(2);
    CT_register_op(ctx, OP_CONT, contiguous_handler);
    CT_register_op(ctx, OP_INDEX, index_handler);
    
    
    int shape[2] = {3, 4};
    tensor t = tensor_build(2, shape, sizeof(float), NULL, NULL);


    op_args index_args = malloc(sizeof(tensor*) + sizeof(int*));
    ((tensor**) index_args)[0] = &t;

    int idxs[6] = {0, 2, 1,   0, 2, 1};
    ((int**)(index_args + sizeof(tensor*)))[0] = idxs;
    

    // op cont_op = {
    //     .args=&t, .op_id = OP_CONT, .next = NULL, .prev = NULL

    // };

    // op index_op = {.args=index_args,.op_id=OP_INDEX,.next = &cont_op,.prev = NULL };

    tensor t2 =    index_handler(ctx, index_args);
    tensor t3 = contiguous_handler(ctx, &t2);
    
    tensor_print_meta(t);
    tensor_print_meta(t2);
    tensor_print_meta(t3);

    return 0;

}



