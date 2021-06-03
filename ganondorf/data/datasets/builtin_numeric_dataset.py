import os
from typing import Tuple, ClassVar
import attr
from tensorflow_federated.python.simulation import ClientData
from . import federated
from . import medical

__BUILTIN_DATASET_CLIENT_ID_RANGE = \
  {
    "arcm"    : (1, 16), # 15 clients
    "har"     : (1, 7),  #  6 clients (actually 6 classes used as clients)
    "mhealth" : (1, 11) # 10 clients
  }

def load_numeric(dataset_name: str,
         normalised: bool = True) -> federated.Federated_Loader:
  """ Loads a builtin numeric (array like) dataset

  Parameters
  ----------
  dataset_name : str
    The name of the dataset

  normalised : bool, deafult=True
    Whether or not the loaded dataset is normalised or raw

  Returns
  -------
  federated.Federated_Loader
    A Federated Loader that loads the requested dataset

  """
  return _DatasetLoader(
    filename = f'{dataset_name}_data{"_norm" if normalised else ""}.npz',
    directory = f"{os.path.dirname(__file__)}/datasets/{dataset_name}",
    client_id_range = __BUILTIN_DATASET_CLIENT_ID_RANGE[dataset_name]
    )


@attr.s(auto_attribs=True, slots=True, frozen=True)
class _DatasetLoader(federated.Federated_Loader):
  filename: str
  directory: str
  client_id_range: Tuple[int, int]

  client_id_prefix: ClassVar[str] = "client_"
  client_train_suffix: ClassVar[str] = "_train_data"
  client_test_suffix: ClassVar[str] = "_test_data"

  def load_dataset(self) -> Tuple[ClientData, ClientData]:
    return self._loader_helper().load_dataset()


  def _loader_helper(self) -> federated.NpzLoader:

    client_ids = [f"{_DatasetLoader.client_id_prefix}{i}"
                  for i in range(*self.client_id_range)]

    return federated.NpzLoader(
        filename = f"{self.directory}/{self.filename}",
        client_ids = client_ids,
        client_train_suffix = _DatasetLoader.client_train_suffix,
        client_test_suffix = _DatasetLoader.client_test_suffix,
        label_index = -1
        )
