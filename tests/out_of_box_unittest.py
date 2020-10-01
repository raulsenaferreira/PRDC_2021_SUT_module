import unittest
import json
import numpy as np
#from novelty_detection.utils import abstraction_box


class TestAbstractionBox(unittest.TestCase):
    

    def test_make_box(self, data):
        data = np.asarray(data)

        data = data[:,[0,-1]]
        self.assertEqual(data[0, 0], -2, "Should be -2")
        self.assertEqual(data[0, 1], 1, "Should be 1")
        

    #def test_sum_tuple(self):
        #abstraction_box.make_abstraction(data, K, classe, dim_reduc_obj=None, dim_reduc_method='', monitors_folder='', save=False)
        #self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()
