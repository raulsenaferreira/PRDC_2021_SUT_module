import json
from src.tests.out_of_box_unittest import TestAbstractionBox
from src.tests.fixtures import mock_data

data = mock_data.load_array()
test = TestAbstractionBox()
test.test_make_box(data)