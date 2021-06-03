""" DOC STRING """
from abc import ABC, abstractmethod
from typing import Tuple, Iterable #, Mapping
import attr
import numpy as np
from tensorflow_federated.python.simulation import ClientData
from . import loader

class FederatedLoader(ABC):
  """ Interface for loading a tensorflow federated database.

  """

  @abstractmethod
  def load_dataset(self) -> Tuple[ClientData, ClientData]:
    pass

# class MappingLoader(Loader):
#   """ Interface

@attr.s(auto_attribs=True, slots=True, frozen=True)
class NpzLoader(FederatedLoader):
  """ Loader for importing Numpy NPZ files (Compressed or not).
  The names of the included arrays should be the names for the clients
  used in a federated context with information on whether it is training or
  testing data although this is flexible

  """

  filename: str
  client_ids: Iterable[str]
  client_train_suffix: str = ""
  client_test_suffix: str = ""
  label_index: int = -1

  load_test_data: bool = True

  def load_dataset(self) -> Tuple[ClientData, ClientData]:

    mapping_dataset = np.load(self.filename)
    train_dataset = {}
    test_dataset = {}

    for client_id in self.client_ids:
      train_data = mapping_dataset[client_id + self.client_train_suffix]

      train_dataset[client_id] = \
          loader.ArrayLoader(train_data, self.label_index).load_dataset()

      if self.load_test_data:
        test_data = mapping_dataset[client_id + self.client_test_suffix]

        test_dataset[client_id] = \
            loader.ArrayLoader(test_data, self.label_index).load_dataset()


    train_client_data = ClientData.from_clients_and_fn(
        self.client_ids,
        train_dataset.get
        )

    if self.load_test_data:
      test_client_data = ClientData.from_clients_and_fn(
          self.client_ids,
          test_dataset.get
          )


    return train_client_data, test_client_data



