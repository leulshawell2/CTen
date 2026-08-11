#include <python3.14/Python.h>
#include "../../backend/cpu/cpu.h"

/**
 * This is the tensor class
 */
typedef struct{
    PyObject_HEAD;
    tensor t;
} Tensor;



static PyTypeObject Tensor_Object;

//python object stuff
static PyObject* Py_Tensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    Tensor* self = (Tensor*)type->tp_alloc(type, 0);
    if(self == NULL){
        return NULL;

    }
    return (PyObject*)self;
}


    
static int Py_Tensor_init(PyObject* self, PyObject* args, PyObject* kwargs){
    return 0;
}



static void PY_Tensor_dealloc(PyObject* self){
    tensor t = ((Tensor*)self)->t;
    tensor_free(&t);
    Py_TYPE(self)->tp_free((PyObject*)self);
}



//tensor ops
static PyObject* Py_Tensor_shape(PyObject* self, PyObject* args) {
    Tensor* other_tensor;
    
    if (!PyArg_ParseTuple(args, "O", &other_tensor)) {
        return NULL;
    }
    
    return PyObject_CallObject((PyObject*)&Tensor_Object, NULL);
}

static PyObject* Py_Tensor_strides(PyObject* self, PyObject* args) {
    Tensor* other_tensor;
    
    if (!PyArg_ParseTuple(args, "O", &other_tensor)) {
        return NULL;
    }
    
    return PyObject_CallObject((PyObject*)&Tensor_Object, NULL);
}

