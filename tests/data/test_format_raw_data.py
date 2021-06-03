import unittest
import numpy as np
import importlib.resources as res
import ganondorf.data.format_raw_data as mut #mut = module under test
from . import data

class TestFormatRawData(unittest.TestCase):

  def test_csv_line_to_typed_array(self):
    expected = np.arange(8).astype(np.float32)

    csv_line = map(str, range(8))
    actual = mut.csv_line_to_typed_array(csv_line, 
                                            ptype=int,
                                            dtype=np.float)

    self.assertTrue(np.all(expected == actual))


  def test_csv_to_array(self):
    self.assertTrue(True)
