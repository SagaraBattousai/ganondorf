#ifndef __ARRAY_FUNCTIONS_H__
#define __ARRAY_FUNCTIONS_H__


#ifndef MODULE_ARRAY_API_NAME
#define MODULE_ARRAY_API_NAME UNNAMED_AND_WILL_PROBABLY_CAUSE_ERROR
#endif

#include <stdarg.h>

typedef void (*mappable_function)(void *, va_list args);

PyObject *map_array(PyObject *arr, mappable_function func, ...);

PyObject *sum_array(PyObject *arr, int rtype);

//PyObject *nearest_convolve(PyObject *arr);

#endif