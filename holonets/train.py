#!/usr/bin/env python

import numpy as np

class Train:
    """
    Iterator class that runs theano functions over data while gathering
    the resulting monitoring values for plotting.
    """
    def __init__(self, channels, n_batches={'Train':1, 
                                            'Validation':1, 
                                            'Test':1}):
        """
        Channels are passed as list or tuple "channels" of dictionaries.
        Expecting each channel as a dictionary with the following entries:
        - "names": <Names of channels as tuple of strings>
        - "dataset": <Which dataset to average this value over (one of train, 
        test, validation). Write "None" for values to be evaluated 
        independently at the end of an epoch.> 
        - "eval": <Theano function to evaluate, expecting it to take an iteger 
        index to slice into a shared variable dataset>
        - "dimensions": <value dimension as string or Holoviews Dimension>
        Also, need to know how many batches are in the dataset to iterate over:
        - n_batches - number of batches in the dataset, as above, as dictionary.
        """
        self.n_batches = n_batches
        # make a dictionary of channel:[dimension]
        self.dimensions = {}
        for channel in channels:
            dimension = channel.get('dimensions',False)
            if dimension:
                self.dimensions[channel['names']] = enforce_iterable(dimension)
        
        # store channels
        self.channels = channels

    def __iter__(self):
        return self

    def next(self):
        self.collected_channels = {}
        # iterate over train, validation and test channels:
        for dataset_name in ['Train', 'Validation', 'Test']:
            # gather right channels for this dataset
            channels = [channel for channel in self.channels 
                    if channel['dataset'] == dataset_name]
            # check we have some channels to iterate over
            if channels != []:
                for i in range(self.n_batches[dataset_name]):
                    # on each batch, execute functions for training channels and
                    # gather results
                    for channel in channels:
                        returned_vals = enforce_iterable(channel['eval'](i))
                        # match them to channel names
                        for name, rval in zip(channel['names'],returned_vals):
                            if not self.collected_channels.get(name, False):
                                self.collected_channels[name] = []
                            self.collected_channels[name].append(rval)
                # take the mean over this epoch for each channel
                for channel in channels:
                    for name in channel['names']:
                        self.collected_channels[name] = \
                                np.mean(self.collected_channels[name])
        # finally, gather the independent channels
        channels = [channel for channel in self.channels 
                    if channel['dataset'] == 'None']
        for channel in channels:
            # assume the function requires no input (could be useful to add 
            # inputs later)
            returned_vals = enforce_iterable(channel['eval']())
            for name, rval in zip(channel['names'], returned_vals):
                self.collected_channels[name] = rval
                
        return self.collected_channels

def enforce_iterable(foo):
    """
    Function to make sure anything passed to it is returned iterable.
    """
    if not hasattr(foo, '__iter__'):
        foo = [foo]
    return foo
