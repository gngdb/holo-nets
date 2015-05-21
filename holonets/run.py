#!/usr/bin/env python

import sys
import holoviews as hv

class EpochLoop:
    """
    Runs a loop over a train object for a given number of epochs.
    """
    def __init__(self, train, dimensions={}):
        """
        Initialisation arguments:
        - train - training iterator object. Assuming this returns a dictionary
        with channel names as keys and float values.
        - dimensions - optional, pass dimensions as a dictionary
        of {<Channel Name>:[<Dimension Name/Holoviews Dimension>]}.
        """
        self.train = train
        self.dimensions = dimensions
        self.i = 0
        self.results = {}

    def run(self, N_epochs, verbose=False):
        """
        Runs the loop for N_epochs, returns a HoloMap of the results.
        Call again to continue from the previous position.
        """
        if verbose:
            print_verbose_intro(N_epochs)

        # if restarting want to go for N_epochs more
        N_epochs += self.i

        for epoch in self.train:
            self.i += 1

            for channel in [ch for ch in epoch.keys() if ch != 'number']:
                # store results in lists in dictionaries:
                try:
                    self.results[channel].append((self.i, epoch[channel]))
                except KeyError:
                    self.results[channel] = [(self.i, epoch[channel])]

            if verbose:
                print_dot()
            if self.i >= N_epochs:
                break

        # turn the dictionary into a HoloMap
        holo_results = self._make_holomap(self.results)
        
        return holo_results

    def _make_holomap(self, dict):
        """
        Takes a dictionary, making assumptions about its contents and turns it 
        into a HoloMap to return for plotting.
        """
        holomap = hv.HoloMap(key_dimensions=['Channel'])
        for channel in dict.keys():
            # check for dimensions for this channel
            value_dimensions = self.dimensions.get(channel, ['Unknown'])
            # for each channel add Curve entry to the HoloMap
            holomap[(channel)] = hv.Curve(dict[channel], 
                    key_dimensions=['Epoch'],
                    value_dimensions=value_dimensions)

        return holomap


def print_verbose_intro(N):
    """
    For verbose mode, indicate length of run with a dot for each epoch.
    """
    print("When there are as many dots below as above the run will be"
            " complete.")
    for i in range(N):
        sys.stdout.write(".")
    sys.stdout.write("\n\n")
    sys.stdout.flush()

    return None

def print_dot():
    sys.stdout.write(".")
    sys.stdout.flush()
