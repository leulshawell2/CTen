#include "./impl/_cpu_impl.h"

/**
 * this kernels assum casting already done. 
 * If the two tensors have different dtypes you should first handles casting
 * broadcasting is handles here though
 */


void tensor_add(context* ctx, op_args args);
void tensor_sub(context* ctx, op_args args);
