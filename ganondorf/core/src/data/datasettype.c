#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <arrayfunctions.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API_NAME

#include <numpy/arrayobject.h>
#include <datasettype.h>

static PyObject *Dataset_sum(PyObject *unused, PyObject *args)
{
  PyObject *sum;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &sum))
  {
    return NULL;
  }

  sum = sum_array(sum, NPY_LONG);

  if (sum == NULL)
  {
    return NULL;
  }

  return sum;
}

static PyMethodDef Dataset_methods[] = {

  {"sum", (PyCFunction)Dataset_sum, METH_VARARGS | METH_STATIC},
  {NULL} // Sentinel
};

PyTypeObject PyDataset_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "data.Dataset",
  .tp_doc = "Dataset Static Class",
  .tp_basicsize = sizeof(PyDatasetObject),
  .tp_itemsize = 0,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_new = PyType_GenericNew,
  .tp_methods = Dataset_methods,
};