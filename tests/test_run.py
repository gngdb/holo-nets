#!/usr/bin/env python

############################
# Tests for the run module #
############################

import unittest
import pytest
import numpy as np

from .context import holonets

class TestHolonetsRun(unittest.TestCase):
    """
    To test that the run module can take a training function, run it over 
    epochs and accumulate results in a HoloMap object.
    """
    @classmethod
    def setUpClass(self):
        # create mock reqs
        self.channel_name = 'Dummy Channel'
        self.train_iterator = DummyIterator(self.channel_name)

        self.train_loop = holonets.run.EpochLoop(self.train_iterator)

    def test_iterate(self):
        """
        Attempt to run training loop for a given number of epochs.
        """
        N_epochs = 10
        test_holomap = self.train_loop.run(10)
        # check holomap contains the right data
        expected = np.array([np.arange(1,11),np.ones(10)]).T
        assert np.allclose(test_holomap[(self.channel_name)].data, expected)

class DummyIterator:
    def __init__(self, channel_name):
        self.channel_name = channel_name
        self.number = 0

    def __iter__(self):
        return self

    def next(self):
        self.number += 1
        return {self.channel_name: 1,
                'number':self.number}


def main():
    unittest.main()

if __name__ == "__main__":
    main()
