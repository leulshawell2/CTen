#include "./core/tensor.h"


#define CT_ERROR(c, sc, ctx)  (ctx)->err = c; (ctx)->sub_err = sc;


#define EXTRACT_ARG_PTR(type, args) *((type*)args)

//define some major and sub errors for the api
//major errors
#define OP_REG_ERR    MAJOR_ERR_END + 1

//sub errors
#define OP_FOUND_ERR  SUB_ERR_END + 1
#define OP_NOTFOUND_ERR  SUB_ERR_END + 2



#define BASE_OPS_COUNT 10

#define OP_CONT 0
#define OP_INDEX 1
#define OP_COPY 2
#define OP_CLONE 3

typedef struct context context; 

typedef void** op_args;
typedef void(*op_handler)(context*, op_args);

typedef struct{
    void* args;
    int op_id;
    struct op* next;
    struct op* prev;
}op;

struct context{

    int mem_usage;
    int tensor_count;
    int err;
    int sub_err;
    
    op_handler ops_table[];
};




/**
 * this create a context for you.
 * holds the entire program context
 */
context* CT_context_init(int op_count);

/**
 * register an op. Ids  are used as indexes in a table
 */

void  CT_register_op(context* ctx, int op_id, op_handler handler);

/**
 * replace an op in the table
 */
void  CT_replace_op(context* ctx, int op_id, op_handler handler);