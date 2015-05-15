#!/usr/bin/env python

############################
# Tests for the run module #
############################

import unittest
import mock
import pytest
import numpy as np

from .context import holonets.run

class TestHolonetsRun(unittest.TestCase):
    """
    To test that the run module can take a training function, run it over 
    epochs and accumulate results in a HoloMap object.
    """
    @classmethod
    def setUpClass(self):
        # create mock reqs
        self.channel_name = 'Dummy Channel'
        self.train_function = mock.MagickMock(return_value={
            self.channel_name: 1 
            })

        self.train_loop = holonets.run.EpochLoop(self.train_function)

    def test_iterate(self):
        """
        Attempt to run training loop for a given number of epochs.
        """
        N_epochs = 10
        test_holomap = self.train_loop.run(10)
        # check holomap contains the right data
        expected = np.array([np.arange(1,11),np.ones(10)]) 
        assert test_holomap[self.channel_name].data == expected

def main():
    unittest.main()

if __name__ == "__main__":
    main()
