#define OMP

#include "./examples/attention.c"

int main(){

    attention_config cfg = { .seq_len=8, .emb_dim=4, .att_dim=12, .att_heads=2};

    tensor wq;
    tensor wk;
    tensor wv;
    tensor input;

    int qkv_shape[3] = {cfg.att_heads, cfg.emb_dim, cfg.att_dim};

    tensor_build(3, qkv_shape, sizeof(float), Float32, NULL, NULL, &wq);
    tensor_build(3, qkv_shape, sizeof(float), Float32, NULL, NULL, &wk);
    
    qkv_shape[2] = cfg.emb_dim;
    tensor_build(3, qkv_shape, sizeof(float), Float32, NULL, NULL, &wv);

    int input_shape[2] = {cfg.seq_len, cfg.emb_dim};
    tensor_build(2, input_shape, sizeof(float), Float32, NULL, NULL, &input);


    attention(cfg, &wq, &wk, &wv, &input);

}






