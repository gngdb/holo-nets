#!/usr/bin/env python



class Train:
    """
    Iterator class that runs theano functions over data while gathering
    the resulting monitoring values for plotting.
    """
    def __init__(self, *channels):
        """
        Expecting each channel as a dictionary with the following entries:
        - "name": <Name of channel as string>
        - "dataset": <Which dataset to average this value over (one of train, 
        test, validation). Write "None" for values to be evaluated 
        independently at the end of an epoch.> 
        - "eval": <Theano function to evaluate, expecting it to take an iteger 
        index to slice into a shared variable dataset>
        - "dimensions": <value dimension as string or Holoviews Dimension>
        """
        # make a dictionary of channel:[dimension]
        self.dimensions = {}
        for channel in channels:
            dimension = channel.get('dimensions',False)
            if dimension:
                self.dimensions[channel['name']] = [dimension]


    def __iter__(self):
        return self

    def next(self):
        pass
