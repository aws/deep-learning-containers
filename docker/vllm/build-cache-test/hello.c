#include <Python.h>
static PyObject* hello(PyObject* self, PyObject* args) { return PyUnicode_FromString("hello from cached wheel"); }
static PyMethodDef methods[] = {{"hello", hello, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "hello_ext", NULL, -1, methods};
PyMODINIT_FUNC PyInit_hello_ext(void) { return PyModule_Create(&module); }
