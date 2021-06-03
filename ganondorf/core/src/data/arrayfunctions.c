#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <arrayfunctions.h>
#include <stdarg.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API_NAME

#include <numpy/arrayobject.h>


PyObject *sum_array(PyObject *arr, int rtype)
{
  PyObject *sum;

  if (PyArray_NDIM(arr) == 1)
  {
    PyArrayObject *input[1] = { arr };

    sum = PyArray_EinsteinSum("i->", 1, input, PyArray_DescrNewFromType(rtype), NPY_KEEPORDER, NPY_SAFE_CASTING, NULL);

    return PyLong_FromLong(*((long *)PyArray_GETPTR1(sum, 0)));
  }
  else
  {
    return PyArray_Sum(arr, NPY_MAXDIMS, rtype, NULL);
  }
}


PyObject *map_array(PyObject *arr, mappable_function func, ...)
{
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;

  /* Handle zero-sized arrays specially */
  if (PyArray_SIZE(arr) == 0) {
    return 0;
  }

  iter = NpyIter_New(arr, 
    NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
    NPY_KEEPORDER, 
    NPY_SAFE_CASTING,
    NULL);
  if (iter == NULL) {
    return -1; //Or Null?
  }

  /*
   * The iternext function gets stored in a local variable
   * so it can be called repeatedly in an efficient manner.
   */
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    return -1;
  }
  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);
  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);
  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  va_list args;
  va_start(args, func);

  do {
    /* Get the inner loop data/stride/count values */
    char *data = *dataptr;
    npy_intp stride = *strideptr;
    npy_intp inner_loop_size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (inner_loop_size--) {
      func((void *)data, args);
      data += stride;
    }

    /* Increment the iterator to the next inner loop */
  } while (iternext(iter));

  va_end(args);
  NpyIter_Deallocate(iter);

  return arr;
}