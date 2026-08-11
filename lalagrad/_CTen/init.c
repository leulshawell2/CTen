/**
 * initilize the context for lalagrad
 */

#include "ops.c"

context* cpu_ctx;

void CT_init(){
    cpu_ctx = CT_context_init(BASE_OPS_COUNT);
    context* cpu_ctx = CT_context_init(2);

    printf("Registering %d ops in the CPU Context", BASE_OPS_COUNT);


    CT_register_op(cpu_ctx, OP_CONTIGUOUS, tensor_contiguous);
    CT_register_op(cpu_ctx, OP_INDEX, tensor_index);
    CT_register_op(cpu_ctx, OP_TRASPOSE, tensor_transpose);
    CT_register_op(cpu_ctx, OP_PERMUTE, tensor_permute);
    CT_register_op(cpu_ctx, OP_ADD, tensor_add);
    CT_register_op(cpu_ctx, OP_SUB, tensor_sub);
    CT_register_op(cpu_ctx, OP_CAST, tensor_cast);


}


/* Module definition */
static PyMethodDef Py_methods[] = {
    {NULL} /* Sentinel */
};

static struct PyModuleDef _C = {
    PyModuleDef_HEAD_INIT,
    .m_name = "CTen",
    .m_doc = "CTen python api",
    .m_size = -1,
    .m_methods = Py_methods,
};



PyMODINIT_FUNC PyInit_CTen(void) {
    PyObject *module;

    CT_init();
    

    if (PyType_Ready(&Op_Object) < 0)
        return NULL;

    module = PyModule_Create(&_C);
    if (module == NULL)
        return NULL;

    Py_INCREF(&Op_Object);
    if (PyModule_AddObject(module, "Ops", (PyObject *)&Op_Object) < 0) {
        Py_DECREF(&Op_Object);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}