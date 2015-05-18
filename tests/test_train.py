#!/usr/bin/env python

##############################
# Tests for the train module #
##############################

import unittest
import mock
import pytest
import numpy as np

from .context import holonets

class TestHolonetsTrain(unittest.TestCase):
    """
    Test to ensure that the train class is able to take some Theano
    functions and run them to evaluate some channels and return the values
    in a dictionary.
    """
    @classmethod
    def setUpClass(self):
        # set up rng
        rng = np.random.RandomState(42)
        self.rval = rng.randn(4)
        # make mock Theano functions
        fprop_train = mock.MagicMock(return_value=(self.rval[0],self.rval[1]))
        fprop_test = mock.MagicMock(return_value=self.rval[2])
        independent = mock.MagicMock(return_value=self.rval[3])
        # train function expects to take these as dictionary with
        # entries to indicate whether to run on train, test or eval 
        # independently after running an epoch
        # Also, should contain the name of the channel
        self.train_channel = {
                'names':('Train Channel','Train Channel 2'),
                'dataset':'Train',
                'eval':fprop_train,
                'dimensions':['Loss', 'Misc']
        }
        self.test_channel = {
                'names':('Test Channel',),
                'dataset':'Test',
                'eval':fprop_test
        }
        self.independent = {
                'names':('Independent Channel',),
                'dataset':'None',
                'eval':independent
        }

        # assuming *args interface
        self.train = holonets.train.Train(
               [self.train_channel, 
                self.test_channel, 
                self.independent]
        )

    def test_traincall(self):
        """
        Call the training object once and see if it returns the expected values
        in the right dictionary format.
        """
        # expecting training object to act like an iterator
        output_dict = next(iter(self.train))
        for i,channel in enumerate([
                       'Train Channel',
                       'Train Channel 2',
                       'Test Channel',
                       'Independent Channel']):
            assert np.allclose(output_dict[channel], self.rval[i])

    def test_dimensions(self):
        """
        Check that it's easily to pull the dimensions back out of this object.
        """
        # want to be able to pull dimensions out using the channels name
        dimensions = self.train.dimensions[('Train Channel','Train Channel 2')]
        assert dimensions == ['Loss', 'Misc']

def main():
    unittest.main()

if __name__ == "__main__":
    main()
