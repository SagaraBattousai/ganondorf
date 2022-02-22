import abc
from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar, ClassVar
import numpy as np

Index = tuple[int, int]
Obj = TypeVar('Obj')
Label = TypeVar('Label')

Context = Callable[[Obj, Label, Obj, Label], float]
Coeff = Callable[[Obj, Obj], float]

class ProbabilityMap(abc.ABC):
  @abc.abstractmethod
  def __call__(self, obj, label) -> float:
    pass

  @abc.abstractmethod
  def update(self, obj, label, value):
    pass

  @abc.abstractmethod
  def update_probabilities(self) -> None:
    pass

  @abc.abstractmethod
  def object_iter(self) -> Iterator:
    pass

  @abc.abstractmethod
  def labels(self) -> Sequence:
    pass

  @abc.abstractmethod
  def probabilities(self):
    pass

class ImageBiProbabilityMap(ProbabilityMap):
  POSITIVE_LABEL: ClassVar[int] = 0
  NEGATIVE_LABEL: ClassVar[int] = 1

  def __init__(self, probability_image: np.array):
    self.probability_image = probability_image
    self.next_probability = np.empty_like(probability_image)
    self.object_iterator = None 
    self.labels = [self.POSITIVE_LABEL, self.NEGATIVE_LABEL]

  def probabilities(self):
    return self.probability_image

  def __call__(self, obj, label):
    i, j = obj
    pos_prob = self.probability_image[i][j]
    return pos_prob if label == self.POSITIVE_LABEL else (1 - pos_prob)

  def update(self, obj, label, value):
    i, j = obj
    pos_prob = value if label == self.POSITIVE_LABEL else (1 - value)
    self.next_probability[i][j] = pos_prob

  def update_probabilities(self) -> None:
    self.probability_image = self.next_probability
    self.next_probability = np.empty_like(self.probability_image)

  def object_iter(self) -> Iterator:
    if self.object_iterator is None:
      height, width = self.probability_image.shape[0:2]
      self.object_iterator = [(i, j) for i in range(width) \
                              for j in range(height)]
    return iter(self.object_iterator)

  def labels(self) -> Sequence:
    return self.labels

def q(ai: Obj, aj: Obj, label: Label, labels: list[Label],
      probability: ProbabilityMap, context):
  acc = 0
  for lab in labels:
    c = context(ai, label, aj, lab)
    p = probability(aj, lab)
    acc += c * p

  return acc
def Q(ai: Obj, label: Label, labels: list[Label],
      probability: ProbabilityMap, context: Context, coeff: Coeff):
  acc = 0
  for aj in probability.object_iter():
    c = coeff(ai, aj)
    if c > 0:
      acc += c * q(ai, aj, label, labels, probability, context)
  
  return acc


def relax(probability: ProbabilityMap, context: Context, coeff: Coeff):
  labels = probability.labels
  label_iter = [labels[0]] if len(labels) == 2 else labels
    
  for li in label_iter:
    for ai in probability.object_iter():
      normaliser = sum([probability(ai, lab) * \
                        Q(ai, lab, labels, probability, context, coeff) \
                        for lab in labels])

      value = (probability(ai, li) * \
               Q(ai, li, labels, probability, context, coeff)) / normaliser

      probability.update(ai, li, value)

  probability.update_probabilities()

    

