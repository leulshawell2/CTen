#include "../backend/cpu/impl/_cpu_impl.h"
#include <math.h>

typedef struct {
    int seq_len;
    int emb_dim;
    int att_dim;
    int att_heads;
}attention_config;



/**
 * expects wq and wk to be (att_head X emb_dim X att_dim) and wv (att_head X emb_dim X emb_dim)
 */
void attention(attention_config cfg, tensor* wq, tensor* wk, tensor* wv, tensor* x){
    tensor Q;
    tensor K;
    tensor V;

    tensor xx;

    tensor K_T;
    tensor att_score;


    //repeat the input along the att_dim dimension so (broadcasting)
    int shape[3];
    shape[0] = cfg.att_heads; shape[1] = cfg.seq_len; shape[2] = cfg.emb_dim;

    _tensor_view(x, shape, x->meta.dim, &xx);
    shape[0] = cfg.att_heads; shape[1] = 1; shape[2] = 1;
    _tensor_repeat(&xx, shape, x);
    return;




    _tensor_matmul_float32(x, wq, &Q);
    _tensor_matmul_float32(x, wk, &K);

    _tensor_transpose(&K, 1, 2, &K_T);

    _tensor_matmul_float32(&Q, &K_T, &att_score);

    
    // int sqrt_d = sqrt(wq->meta.dim);

    // tensor sqrt_d_t;
    // tensor_from_scalar(att_score.meta.dim, att_score.meta.shape, sizeof(float), Float32, &sqrt_d, &sqrt_d_t);

    // _tensor_div_float32(&att_score, &sqrt_d_t,  &att_score);

    //softmax

    _tensor_matmul_float32(x, wv, &V);






}