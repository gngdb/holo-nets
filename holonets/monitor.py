#!/usr/bin/env python

# Heavily inspired by the code in the Lasagne MNIST example:
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
#
# Least concrete part of holo-nets, wouldn't recommend using it

import lasagne.updates
import lasagne.objectives
import lasagne.layers
import lasagne.utils
import lasagne.regularization
import theano
import theano.tensor as T
import time
import itertools


class Expressions:
    """
    Function to return channels to be monitored for the train module.
    Input:
        - output_layer - final layer of a Lasagne network.
        - dataset - dataset as a dictionary of shared variables:
            - "X_train" - of type X_tensor_type
            - "y_train" - y_tensor_type
            - "X_valid" - X_tensor_type
            - "y_valid" - y_tensor_type
        - batch_size - number of examples in each batch
        - update_rule - lasagne update rule to use 
        - X_tensor_type - type of tensor to use for X
        - loss_function - loss function to use
        - deterministic - disable all stochastic elements
        - learning_rate - learning rate to use
        - regularisation - regularisation function to apply (for
        lasagne.regularization.l2)
        - extra_loss - arbitrary extra loss function
        - update_utility - arbitrary extra update utility function
        - momentum - use Lasagne's apply_momentum with given momentum
    """
    def __init__(self, output_layer, dataset, batch_size=128, 
            update_rule=lasagne.updates.adam,
            X_tensor_type=T.matrix,
            y_tensor_type=T.ivector,
            loss_function=lasagne.objectives.categorical_crossentropy,
            loss_aggregate=T.mean,
            deterministic=False,
            learning_rate=0.001,
            regularisation=lambda x: 0.,
            extra_loss=0.,
            update_utility=lambda x: x,
            momentum=None):
        self.output_layer = output_layer
        self.dataset = enforce_shared(dataset, X_tensor_type, y_tensor_type)
        self.batch_size = batch_size
        self.update_rule = update_rule

        # Theano variables for minibatches and batch slicing
        self.batch_index = T.iscalar('batch_index')
        self.batch_slice = slice(self.batch_index*batch_size,
                                (self.batch_index+1)*batch_size)
        self.X_batch = X_tensor_type('X')
        self.y_batch = y_tensor_type('y')

        # set up the objective
        self.network_output = lasagne.layers.get_output(self.output_layer, 
                self.X_batch)
        self.deterministic_output = lasagne.layers.get_output(self.output_layer,
            self.X_batch, deterministic=True)
        self.all_params = lasagne.layers.get_all_params(self.output_layer, 
                trainable=True)
        self.loss_train = loss_aggregate(loss_function(self.network_output, 
            self.y_batch)) + sum([regularisation(p) for p in self.all_params]) \
                    + extra_loss
        self.loss_eval = loss_aggregate(loss_function(self.deterministic_output,
            self.y_batch)) + sum([regularisation(p) for p in self.all_params]) \
                    + extra_loss

        # build initial list of updates at initialisation (makes sense right)
        self.updates = update_rule(self.loss_train, self.all_params, 
                learning_rate)
        # update utility functions
        self.updates = update_utility(self.updates)
        if momentum:
            self.updates = lasagne.updates.apply_momentum(updates, 
                    self.all_params, momentum=momentum)

        # initialise empty channels dictionary
        self.channels = {}

        # add timer channel
        self.channels['timer'] = {
                "names": ("Epoch Time",),
                "dataset": "None",
                "eval": Timer(),
                "dimensions": ['seconds']
            }

        # initialise empty channel specifications dictionary
        self.channel_specs = {}

    def add_channel(self, name, dimension, expression, function):
        """
        For adding channels, requires the following:
            - name: name for the channel
            - dimension: dimension the output value will have
            - expression: Theano expression to be evaluated, or just a function
        to return a value when called (in the case of non-Theano channels).
            - function: which function to compile the expression as part of, 
        one of:
                * "train": training function
                * "valid": validation function
                * "test" : test function
                * "none" : assume expression can be evaluated by itself
        """
        # sort into appropriate dictionary
        if not self.channel_specs.get(function, False):
            self.channel_specs[function] = {}
        self.channel_specs[function][name] = dict(
                dimension=dimension,
                expression=expression,
                function=function
                )

    def build_channels(self):
        """
        Returns a list of dictionaries that can be passed to the train
        module. Compiles the specification contained in self.channel_specs
        into various Theano functions.
        """
        # take the channel specs and compile functions for train, valid and 
        # test where required, then add to channels dictionary to return
        self.iter_funcs = {}
        for function in ['train', 'valid', 'test']:
            if self.channel_specs.get(function, False):
                # extract specifications into lists
                expressions = []
                names = []
                dimensions = []
                functions = []
                for name in self.channel_specs[function]:
                    expressions.append(
                        self.channel_specs[function][name]['expression'])
                    names.append(name)
                    dimensions.append(
                        self.channel_specs[function][name]['dimension'])
                # now compile theano function
                if function == 'train':
                    updates = self.updates
                else:
                    updates = {}
                self.iter_funcs[function] = theano.function(
                        [self.batch_index],
                        expressions,
                        updates=updates,
                        givens={
                            self.X_batch: self.dataset['X_'+function][self.batch_slice],
                            self.y_batch: self.dataset['y_'+function][self.batch_slice],
                        } 
                        )
                # and add this to the channels dictionary
                self.channels[function] = dict(
                        names=tuple(names),
                        dataset=function,
                        eval=self.iter_funcs[function],
                        dimensions=dimensions
                        )

        return self.channels.values()
    
    def loss(self, dataset, deterministic, name=None):
        """
        Builds a channel specification for loss, given a dataset on which to 
        monitor it. If deterministic is true, noise or dropout will not be 
        applied.
        - dataset: one of "train", "test", "valid"
        - deterministic: True or False

        If required, also specify custom name.
        """
        if not name:
            if deterministic:
                name="{0} Loss".format(dataset)
            else:
                name="{0} Loss with Dropout".format(dataset)

        if deterministic:
            return dict(
                    name=name,
                    function=dataset,
                    expression=self.loss_eval,
                    dimension="Loss"
                    )
        else:
            return dict(
                    name=name,
                    function=dataset,
                    expression=self.loss_train,
                    dimension="Loss"
                    )


    def accuracy(self, dataset, deterministic, name=None):
        """
        Builds a channel specification for accuracy, given a dataset on which
        to monitor it. If deterministic is true, noise or dropout will not be 
        applied.
        - dataset: one of "train", "test", "valid"
        - deterministic: True or False

        If required, also specify custom name.
        """
        self.pred = T.argmax(
            lasagne.layers.get_output(self.output_layer, self.X_batch, 
                deterministic=deterministic), axis=1)
        accuracy = T.mean(T.eq(self.pred, self.y_batch), 
                dtype=theano.config.floatX)
        
        if not name:
            if deterministic:
                name="{0} Accuracy".format(dataset)
            else:
                name="{0} Accuracy with Dropout".format(dataset)

        return dict(
                name=name,
                function=dataset,
                expression=accuracy,
                dimension="Accuracy"
                )

    def update_ratio(self, name="Mean Update Ratio"):
        """
        Builds channel specification for monitoring update ratios.
        """
        self.update_ratios = [T.abs_((self.updates[param]-param)/param) 
                for param in self.all_params]
        self.mean_update_ratio = sum(T.mean(p) 
                for p in self.update_ratios)/len(self.update_ratios)
        
        return dict(
                name=name,
                function="train",
                expression=self.mean_update_ratio,
                dimension="update/param"
                )

    def L2(self, name="Parameters L2"):
        """
        Global L2 channel for parameters.
        """
        L2 = sum(T.sum(p**2) for p in self.all_params)
        return dict(
                name=name,
                function="train",
                expression=L2,
                dimension="L2 norm"
                )

def classification_channels(expressions):
    """
    Takes an instance of Expressions and produces a list of channels that 
    are relevant for a classification experiment.

    Can then be applied to Expressions instance with the add_channel method
    in a loop:
    for channel_spec in channel_specs:
        expressions.add_channel(**channel_spec)
    """
    channel_specs = []
    # loss and accuracy with and without dropout on training and validation
    for deterministic,dataset in itertools.product([True, False],
                                                   ["train","valid"]):
        channel_specs.append(expressions.loss(dataset, deterministic))
        channel_specs.append(expressions.accuracy(dataset, deterministic))
    # update ratio
    channel_specs.append(expressions.update_ratio())
    # global L2 norm
    channel_specs.append(expressions.L2())
    return channel_specs

def enforce_shared(dataset, X_tensor_type, y_tensor_type):
    """
    Datasets as dictionaries containing numpy arrays and those containing
    shared variables will both be returned as dictionaries containing shared
    variables.
    """
    for X_name in [n for n in dataset.keys() if 'X' in n]:
        if not issharedvar(dataset[X_name]):
            dataset[X_name] = theano.shared(
                    lasagne.utils.floatX(dataset[X_name]))
    for y_name in [n for n in dataset.keys() if 'y' in n]:
        if y_tensor_type == T.ivector:
            if not isinstance(dataset[y_name], T.TensorVariable):
                dataset[y_name] = T.cast(dataset[y_name].ravel(), 'int32')
        else:
            if not issharedvar(dataset[y_name]):
                dataset[y_name] = theano.shared(
                        lasagne.utils.floatX(dataset[y_name]))
    return dataset

def issharedvar(var):
    """
    Returns true if a variable is a Theano shared variable. Crude hack.
    """
    return isinstance(var, theano.sandbox.cuda.var.CudaNdarraySharedVariable) \
            or isinstance(var, theano.tensor.sharedvar.TensorSharedVariable)

class Timer:
    """
    Returns number of seconds since it was last called.
    """
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        self.then = self.now
        self.now = time.time()
        return self.now - self.then

