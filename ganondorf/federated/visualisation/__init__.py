""" DOC String """
import collections
import statistics
from typing import Tuple, Sequence, Generator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import ClientData as TffClientData

def normal_quantiles(length: int,
                     mu:float=0.0,
                     sigma:float=1.0)-> Generator[float, None, None]:

  norm_dist = statistics.NormalDist(mu=mu, sigma=sigma)
  splits = 1.0 / length
  current = splits
  count = 0
  while count < length:
    if current >= 1.0:
      current = 0.9999999999999999

    yield norm_dist.inv_cdf(current)
    current += splits
    count += 1

def QQ_plot(data, #pylint: disable=C0103
            mu:float=0.0, sigma:float=1.0):
  normal_data = normal_quantiles(len(data) + 1, mu=mu, sigma=sigma)
  x = list(normal_data)[:-1]
  plt.plot(x, data)
  plt.show()




def visualise_class_counts_for_clients(dataset: tf.data.Dataset,
                                       classes: Sequence[int],
                                       figsize: Tuple[int, int] = (12,7)
                                       ) -> None:
  fig = plt.figure(figsize=figsize)
  fig.suptitle("Class Counts for Client")

  plot_data = collections.defaultdict(list)
  for data in dataset:
    label = int(data["y"].numpy()[0])
    plot_data[label].append(label)
  for lab in classes:
    plt.hist(
        plot_data[lab],
        density=False,
        bins=classes)

  plt.show()






















