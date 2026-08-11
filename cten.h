#include "./core/tensor.h"




#define CT_ERROR(c, sc, ctx)  (ctx)->err = c; (ctx)->sub_err = sc;
#define ERROR_FROM_RES(c, t)  (ctx)->err = (t)->meta.err; (ctx)->sub_err = (t)->meta.sub_err; return;




#define EXTRACT_ARG_PTR(type, args) *((type*)args)

//major errors
#define OP_REG_ERR    CORE_MAJOR_ERR_END + 1
#define DTYPE_ERROR   CORE_MAJOR_ERR_END + 2

//sub errors
#define FOUND_ERR     CORE_SUB_ERR_END + 1
#define NOTFOUND_ERR  CORE_SUB_ERR_END + 2




#define OP_CONTIGUOUS   0
#define OP_CLONE        1
#define OP_COPY         2
#define OP_VIEW         3
#define OP_RESHAPE      4
#define OP_REPEAT       5
#define OP_BROADCAST    6
#define OP_INDEX        7
#define OP_TRASPOSE     8
#define OP_PERMUTE      9
#define OP_ADD          10
#define OP_SUB          11
#define OP_CAST         12

#define BASE_OPS_COUNT 13


typedef struct context context; 

typedef void** op_args;
typedef void(*op_handler)(context*, op_args);

typedef struct{
    void* args;
    int op_id;
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