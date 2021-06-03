#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdarg.h>

#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API_NAME
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#define DATA_MODULE
#include <datamodule.h>
#include <datasettype.h>
#include <arrayfunctions.h>

extern PyTypeObject PyDataset_Type;

void sub(int *value, va_list args)
{
  int subtractor = va_arg(args, int);
  *value = *value - subtractor;
}

void mult(int *value, va_list args)
{
  double multiplier = va_arg(args, double);
  double val = *value;
  *value = floor(val * multiplier);
}


static PyObject *datamodule_window_level(PyObject *self, PyObject *args)
{
  PyObject *image;
  int window;
  int level;

  if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &image, &window, &level))
  {
    return NULL;
  }

  int half_window = window / 2;
  int window_min = level - half_window;
  int window_max = level + half_window;

  image = PyArray_Clip(image, PyLong_FromLong(window_min), PyLong_FromLong(window_max), NULL);
  map_array(image, &sub, window_min);

  PyObject *max_scalar = PyArray_Max(image, NPY_MAXDIMS, NULL);
  int max_value = PyArray_PyIntAsInt(max_scalar);

  double scale = 255.0 / ((double)max_value);
  
  map_array(image, &mult, scale);

  return image;
}






static PyMethodDef datamodule_methods[] = {
  {"window_level", (PyCFunction)datamodule_window_level, METH_VARARGS, "doc string"},
  {NULL} // Sentinel
};



static PyModuleDef datamodule = {
  PyModuleDef_HEAD_INIT,
  .m_name = "data",
  .m_doc = "data module doc string",
  .m_size = -1,
  .m_methods = datamodule_methods
};

PyMODINIT_FUNC
PyInit_datamodule(void)
{
  PyObject *m;
  if (PyType_Ready(&PyDataset_Type) < 0)
  {
    return NULL;
  }

  m = PyModule_Create(&datamodule);
  if (m == NULL)
  {
    return NULL;
  }

  import_array();
  //import_umath();

  Py_INCREF(&PyDataset_Type);
  if (PyModule_AddObject(m, "Dataset", (PyObject *)&PyDataset_Type) < 0)
  {
    Py_DECREF(&PyDataset_Type);
    Py_DECREF(m);
    return NULL;
  }

  return m;

}















/*
static PyObject* ganondorf_add(PyObject *self, PyObject *args)
{
  const int a;
  const int b;

  if (!PyArg_ParseTuple(args, "ii", &a, &b))
  {
    return NULL;
  }

  int c = a + b;

  return PyLong_FromLong(c);

}


static PyMethodDef GanondorfMethods[] = {
  {"gadd", ganondorf_add, METH_VARARGS, "adds"},


  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ganondorfmodule = {
  PyModuleDef_HEAD_INIT,
  "gdc",
  NULL, //Documentation
  -1,
  GanondorfMethods
};

PyMODINIT_FUNC PyInit_ganondorf_core(void)
{
  return PyModule_Create(&ganondorfmodule);
}



*/
