""" Functional Module for functional programming to
imporve generic programming """

import functools
from typing import Iterable, List, Callable, TypeVar

T = TypeVar("T")

def flat_map_in_out_specs(fns: Iterable[Callable[[int, int], T]],
                          input_shape: int,
                          classes: int) -> List[Callable[[], T]]:

  return list(map(
      lambda a: \
          functools.partial(a, input_shape=input_shape, classes=classes),
          fns))


#def model_fn_from_spec
