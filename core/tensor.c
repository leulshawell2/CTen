
#include  "tensor.h"


//some helpers i don't want to be function calls
#define tensor_isshared(t)  *((t)->meta.ref) != 0
#define tensor_fromother(t) tensor_build((t)->meta.dim, (t)->meta.shape, (t)->meta.e_size, None, &t, NULL)


void tensor_print_data(tensor* t){
    tensor_meta m = t->meta;

    switch (t->meta.dtype)
    {
    case Int32:
        for (uint i = 0; i < m.size; i++) {
            int i1 = 0;
            for(uint8 d=0; d < m.dim; d++){
                int coord = GET_MIXED_RADIX_DIGIT(i, d, m.__stride, m.shape);
                i1 += (coord * m.stride[d]);
            }

            printf("%d,\t", ((int*)t->data)[i1]);
        }
        break;
    default:
        for (uint i = 0; i < m.size; i++) {
                int i1 = 0;
                for(uint8 d=0; d < m.dim; d++){
                    int coord = GET_MIXED_RADIX_DIGIT(i, d, m.__stride, m.shape);
                    i1 += (coord * m.stride[d]);
                }

                printf("%f,\t", ((float*)t->data)[i1]);
            }
            break;
        }
    printf("\n");

}




void tensor_build(uint8 dim, int* shape, uint8 e_size, uint8 dtype, tensor* pr, void* data, tensor* res){
    
    res->meta.err = 0;
    res->meta.sub_err = 0;
    res->meta.dim = dim;
    res->meta.e_size = e_size;
    res->meta.dtype = dtype;
    
    int size = 1;
    for(uint8 d=0; d < dim; d++) 
        size = size * shape[d];
    res->meta.size = size;
    
    
    int meta_size = sizeof(int) * dim;
    void* meta = meta_alloc(dim);
    if(!meta){
        ERROR(MEM_ERR, MALLOC_ERR, res)
    }
    
    
    //build the meta block    
    memcpy(meta, shape, meta_size);
    res->meta.shape = meta;
    res->meta.stride = meta + meta_size;
    res->meta.__stride = res->meta.stride + dim;
    res->meta.ref   = res->meta.__stride + dim;

    res->meta.stride[dim-1] = 1;
    for (int8 d = dim-2; d > -1; d--)
        res->meta.stride[d] = shape[d+1] * res->meta.stride[d+1];


    memcpy(res->meta.__stride, res->meta.stride, meta_size);
    
    if(pr){
        //create from other tensor
        *(pr->meta.ref) += 1;
        res->meta.ref = pr->meta.ref;
        res->data = pr->data;
        res->meta.dtype = dtype? dtype: pr->meta.dtype;
    }else if(data){
        //create from void* data
        res->data = data;
    }else{
        res->data = malloc(size * e_size);
        if(res->data == NULL){
            ERROR(MEM_ERR, MALLOC_ERR, res)
        }
    }
    
}


void tensor_free(tensor* t){
    if(tensor_isshared(t))
        *(t->meta.ref ) -= 1;
    else
        free(t->data);
    free(t->meta.shape);
}



void tensor_print_meta(tensor* t){
        printf("Addr=%p Dtype=%s ", t->data, t->meta.dtype==Float32? "f32": "i32");
        meta_print(&t->meta);

}



boolean tensor_iscontiguous(tensor* t){
    return (memcmp(t->meta.stride, t->meta.__stride, t->meta.dim) == 0);
}


int tensor_size(tensor* t){
    return meta_size(t->meta.shape, t->meta.dim);
}




void tensor_meta_clone(tensor* dest, tensor* src){
    dest->meta.dim = src->meta.dim;
    dest->meta.e_size = src->meta.e_size;
    dest->meta.size = src->meta.size;
    dest->meta.sub_err = src->meta.sub_err;
    dest->meta.err = src->meta.err;
    memcpy(dest->meta.shape, dest->meta.shape, sizeof(int) * src->meta.dim * 3 + sizeof(int*));
}