#include "tensor.c"




/**
 * In python space you work with the Op mostly
 * until you call .realize() that is when all ops run and you get the tensor
 */
typedef struct{
    PyObject_HEAD;
    tensor* t;
    op op;
} Op;



static PyTypeObject Op_Object;

//python object stuff
static PyObject* Py_Op_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    Op* self = (Op*)type->tp_alloc(type, 0);
    if(self == NULL)
        return Py_None;
    
    return (PyObject*)self;
}



static int Py_Op_init(PyObject* self, PyObject* args, PyObject* kwargs){
    ((Op*)self)->op.next = NULL;
    ((Op*)self)->op.prev = NULL;
    ((Op*)self)->t = NULL;
    return 0;
}



static void PY_Op_dealloc(PyObject* self){
    Py_TYPE(self)->tp_free((PyObject*)self);
}



//tensor ops
static PyObject* Py_Op_repeat(PyObject* self, PyObject* args) {
    Op* other_tensor;
    
    if (!PyArg_ParseTuple(args, "O", &other_tensor)) {
        return NULL;
    }
    
    return PyObject_CallObject((PyObject*)&Op_Object, NULL);
}

static PyObject* Py_Op_add(PyObject* self, PyObject* args) {
    Op* other_tensor;
    
    if (!PyArg_ParseTuple(args, "O", &other_tensor)) {
        return NULL;
    }
    
    return PyObject_CallObject((PyObject*)&Op_Object, NULL);
}