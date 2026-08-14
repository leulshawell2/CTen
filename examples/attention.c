#include "../backend/cpu/impl/_cpu_impl.h"
#include <math.h>

typedef struct {
    int seq_len;
    int emb_dim;
    int att_dim;
    int att_heads;
}attention_config;



void element_sqrt(float* e1, float*e2){
    *((float*)e1) = sqrt(*((float*)e2));
}




/**
 * expects wq and wk to be (att_head X emb_dim X att_dim) and wk (att_head X emb_dim X emb_dim)
 */
void attention(attention_config cfg, tensor* wq, tensor* wk, tensor* wv, tensor* x){
    tensor Q;
    tensor K;
    tensor V;


    int d = wq->meta.shape[wq->meta.dim - 2];

    tensor K_T;
    tensor att_score;

    _tensor_matmul_float32(x, wq, &Q);
    _tensor_matmul_float32(x, wk, &K);

    _tensor_transpose(&K, 1, 2, &K_T);

    _tensor_matmul_float32(&Q, &K_T, &att_score);
    _tensor_map(&att_score, element_sqrt, &att_score);

    //softmax

    



    _tensor_matmul_float32(x, wv, &V);






}