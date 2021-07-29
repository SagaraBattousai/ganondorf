#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdarg.h>

#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API_NAME
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#define DATA_MODULE
#include <datacore.h>
#include <arrayfunctions.h>

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

static PyObject *datacore_window_level(PyObject *self, PyObject *args)
{
  PyArrayObject *image;
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

static PyMethodDef datacore_methods[] = {
  {"window_level", (PyCFunction)datacore_window_level, METH_VARARGS, "doc string"},
  {NULL} // Sentinel
};

static PyModuleDef datacore = {
  PyModuleDef_HEAD_INIT,
  .m_name = "data",
  .m_doc = "Core data module doc string",
  .m_size = -1,
  .m_methods = datacore_methods
};

PyMODINIT_FUNC
PyInit_datacore(void)
{
  PyObject *module = PyModule_Create(&datacore);
  if (module == NULL)
  {
    return NULL;
  }

  import_array();
  //import_umath();

  return module;

}
