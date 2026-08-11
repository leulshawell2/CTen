#include <python3.14/Python.h>
#include "../../backend/cpu/cpu.h"




/**
 * In python space you work with the Op mostly
 * until you call .realize() that is when all ops run and you get the tensor
 */
typedef struct{
    PyObject_HEAD;
    op* op;
    struct op* next;
    struct op* prev;
} Op;



//create Op object here (python __new__) 
static PyObject* Py_Op_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    Op* self = (Op*)Py_TYPE(type)->tp_alloc(type, 0);
    self->next = NULL;
    self->prev = NULL;
    if(self == NULL)
        return None;
    return (PyObject*)self;
}



//inititlize the Op in here. (python __init__)
static int Py_Op_init(PyObject* self, PyObject* args, PyObject* kwargs){
    return 0;
}



//define destractor (python __delete__) 
static void PY_Op_dealloc(Op* self){
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject* Py_Op_repeat(PyObject* self, PyObject* args){
    PyObject* new =  Py_Op_new(Py_TYPE(self), args, NULL);
    return new;
}




static PyMethodDef Py_Op_Methods[] = {
    {"repeat", Py_Op_repeat, METH_VARARGS},
    {NULL}
};

static PyGetSetDef Py_Op_Props[] = {
    {NULL}
};

static PyTypeObject Op_Object = {
    PyVarObject_HEAD_INIT(NULL, 0)  
    .tp_name = "CTen.Ops",  
    .tp_doc = "The C Ops backend\nThis is the backbone of lalagrad ops\nIt create a graph of ops that get lazilly evaluated",      
    .tp_basicsize = sizeof(Op), 
    .tp_itemsize = 0,                 
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  
    .tp_new = Py_Op_new,           
    .tp_init = (initproc)Py_Op_init, 
    .tp_dealloc = (destructor)PY_Op_dealloc, 
    .tp_methods = Py_Op_Methods,   
    .tp_getset = Py_Op_Props, 
};