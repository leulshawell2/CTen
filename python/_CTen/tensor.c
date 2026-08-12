#include <python3.14/Python.h>
#include "../../backend/cpu/cpu.h"

context* cpu_ctx;


/**
 * In python space you work with the Tensor mostly
 * until you call .realize() that is when all Tensors run and you get the tensor
 */
typedef struct{
    PyObject_HEAD;
    op* last;
} Tensor;



static PyTypeObject Tensor_Object;

//python object stuff
static PyObject* Py_Tensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    Tensor* self = (Tensor*)type->tp_alloc(type, 0);
    if(self == NULL)
        return Py_None;
    
    return (PyObject*)self;
}



static int Py_Tensor_init(PyObject* _self, PyObject* args, PyObject* kwargs){
    Tensor* self = (Tensor*)_self;

    self->last = malloc(200);

    self->last->next = NULL;
    self->last->prev = NULL;
    self->last->completed = False;
    

    int opcode;
    PyObject* shape_tuple;
    int dtype;

    
    if (!PyArg_ParseTuple(args, "iOi", &opcode, &shape_tuple, &dtype)) {
        return -1;
    }
    
    if (!PyTuple_Check(shape_tuple)) {
        PyErr_SetString(PyExc_TypeError, "shape arg must be tuple");
        return -1;
    }
    
    Py_ssize_t dims = PyTuple_Size(shape_tuple);
    int shape[dims];

    for(int d=0; d < dims; d++){
        PyObject* d_size = PyTuple_GetItem(shape_tuple, d);
        PyLong_AsInt32(d_size, shape + d);
    }

    self->last->opcode = opcode;
    
    return 0;
}



static void PY_Tensor_dealloc(PyObject* self){
    Py_TYPE(self)->tp_free((PyObject*)self);
}



//tensor Tensors
static PyObject* Py_Tensor_repeat(PyObject* self, PyObject* args) {
    tensor* input = ((tensor**)((Tensor*)self)->last->args)[0];
    size_t arg_size = sizeof(tensor*) + input->meta.dim * sizeof(int);
    // op* repeat_op = CT_op_build(cpu_ctx, arg_size, OP_REPEAT, NULL, ((Tensor*)self)->last);
    return self;
    
}

static PyObject* Py_Tensor_add(PyObject* self, PyObject* args) {
    return self;
}


static PyObject* Py_Tensor_realize(PyObject* self, PyObject* args){
    // CT_dispatch_graph(cpu_ctx, &((Tensor*)self)->op);
    return self;
}