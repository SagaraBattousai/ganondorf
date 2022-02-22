import tensorflow as tf

__all__ = ['normal_in_range']

@tf.function
def normal_in_range(shape, minimum:float=0, maximum:float=1.0,
                    std_deviations:float=4.5, clip:bool=True,
                    **kwargs)->tf.Tensor:

  dtype = kwargs.get('dtype', tf.dtypes.float32)
  seed = kwargs.get('seed', None)
  name = kwargs.get('name', None)

  stddev = (maximum - minimum) / (2 * std_deviations)
  mean   = (maximum + minimum) / 2

  normaldist = tf.random.normal(shape, mean, stddev, dtype, seed, name)
  if clip:
    normaldist = tf.clip_by_value(normaldist, minimum, maximum)

  return normaldist

