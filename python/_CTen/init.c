/**
 * initilize the context for lalagrad
 */

#include "tensor.c"





//CTen init
void CT_init(){
    cpu_ctx = CT_context_init(BASE_OPS_COUNT);
    context* cpu_ctx = CT_context_init(2);


    CT_register_op(cpu_ctx, OP_CONTIGUOUS, tensor_contiguous);
    CT_register_op(cpu_ctx, OP_INDEX, tensor_index);
    CT_register_op(cpu_ctx, OP_TRASPOSE, tensor_transpose);
    CT_register_op(cpu_ctx, OP_PERMUTE, tensor_permute);
    CT_register_op(cpu_ctx, OP_ADD, tensor_add);
    CT_register_op(cpu_ctx, OP_SUB, tensor_sub);
    CT_register_op(cpu_ctx, OP_CAST, tensor_cast);

}



//Ops Object Init
static PyMethodDef Py_Tensor_Methods[] = {
    {"add", Py_Tensor_add, METH_VARARGS},
    {"repeat", Py_Tensor_repeat, METH_VARARGS},
    {"realize", Py_Tensor_realize, METH_NOARGS},
    {NULL}
};



static PyGetSetDef Ops_Props[] = {
    {NULL}
};

static PyTypeObject Op_Object = {
    PyVarObject_HEAD_INIT(NULL, 0)  
    .tp_name = "CTen.Tensor",  
    .tp_doc = "The C Ops backend\nThis is the backbone of lalagrad ops\nIt create a graph of ops that get lazilly evaluated",      
    .tp_basicsize = sizeof(Tensor), 
    .tp_itemsize = 0,                 
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  
    .tp_new = Py_Tensor_new,           
    .tp_init = (initproc)Py_Tensor_init, 
    .tp_dealloc = (destructor)PY_Tensor_dealloc, 
    .tp_methods = Py_Tensor_Methods,   
    .tp_getset = Ops_Props, 
};




static PyMethodDef Module_Methods[] = {
    {NULL}
};


//Module Init
static struct PyModuleDef CTen = {
    PyModuleDef_HEAD_INIT,
    .m_name = "CTen",
    .m_doc = "CTen python api",
    .m_size = -1,
    .m_methods = Module_Methods
};



PyMODINIT_FUNC PyInit_CTen(void) {
    PyObject *module;

    CT_init();

    if(cpu_ctx->err){
        printf("CTen initlization failed. ERR: %d, SUB_ERR: %d", cpu_ctx->err, cpu_ctx->sub_err);
        return Py_None;
    }
    

    if (PyType_Ready(&Op_Object) < 0)
        return NULL;

        
    module = PyModule_Create(&CTen);
    if (module == NULL)
    return NULL;
        
    if (PyModule_AddObjectRef(module, "Tensor", (PyObject *)&Op_Object) < 0) {
        Py_DECREF(&Op_Object);
        Py_DECREF(module);
        return NULL;
    }
    

    return module;
}

