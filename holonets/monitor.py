#!/usr/bin/env python

# Heavily inspired by the code in the Lasagne MNIST example:
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

import lasagne.updates
import lasagne.objectives
import lasagne.layers
import lasagne.utils
import lasagne.regularization
import theano
import theano.tensor as T
import time

class Expressions:
    """
    Function to return channels to be monitored for the train module.
    Input:
        - output_layer - final layer of a Lasagne network.
        - dataset - dataset as a dictionary of shared variables:
            - "X_train" - tensor
            - "y_train" - ivector 
            - "X_valid" - tensor
            - "y_valid" - ivector
        - batch_size - number of examples in each batch
        - update_rule - lasagne update rule to use 
        - X_tensor_type - type of tensor to use for X
        - loss_function - loss function to use
        - deterministic - disable all stochastic elements
        - learning_rate - learning rate to use
    """
    def __init__(self, output_layer, dataset, batch_size=128, 
            update_rule=lasagne.updates.adadelta,
            X_tensor_type=T.matrix,
            loss_function=lasagne.objectives.categorical_crossentropy,
            deterministic=False,
            learning_rate=0.1):
        self.output_layer = output_layer
        self.dataset = enforce_shared(dataset)
        self.batch_size = batch_size
        self.update_rule = update_rule

        # Theano variables for minibatches and batch slicing
        self.batch_index = T.iscalar('batch_index')
        self.batch_slice = slice(self.batch_index*batch_size,
                                (self.batch_index+1)*batch_size)
        self.X_batch = X_tensor_type('X')
        self.y_batch = T.ivector('y')

        # set up the objective
        self.objective = lasagne.objectives.Objective(self.output_layer, 
                loss_function=loss_function)
        self.loss_train = self.objective.get_loss(self.X_batch, 
                target=self.y_batch, 
                deterministic=deterministic)
        self.loss_eval = self.objective.get_loss(self.X_batch, 
                target=self.y_batch,
                deterministic=True)

        self.pred = T.argmax(
            self.output_layer.get_output(self.X_batch, 
                deterministic=True), axis=1)
        self.accuracy = T.mean(T.eq(self.pred, self.y_batch), 
                dtype=theano.config.floatX)
    
        # build initial list of updates at initialisation (makes sense right)
        self.all_params = lasagne.layers.get_all_params(output_layer)
        self.updates = update_rule(self.loss_train, self.all_params, 
                learning_rate)

        # initialise empty channels list
        self.channels = {}

        # add timer channel
        self.channels['timer'] = {
                "names": ("Epoch Time",),
                "dataset": "None",
                "eval": Timer(),
                "dimensions": ['seconds']
            } 


    def build_channels(self):
        """
        Returns a list of dictionaries that can be passed to the train
        module.
        """
        self.add_default_channels()

        return self.channels.values()
    
    def add_default_channels(self):
        """
        Checks for and adds default channels, if they've not been made yet.
        """
        if not self.channels.get('train', False):
            iter_train = theano.function([self.batch_index], 
                    [self.loss_train, self.accuracy], 
                    updates=self.updates,
                    givens={
                        self.X_batch: self.dataset['X_train'][self.batch_slice],
                        self.y_batch: self.dataset['y_train'][self.batch_slice],
                    },
            )

            self.channels['train'] = {
                "names":("Train Loss","Train Accuracy"),
                "dataset": "Train",
                "eval": iter_train,
                "dimensions": ['Loss', 'Accuracy']
                }

        if not self.channels.get('validation', False):
            iter_valid  = theano.function([self.batch_index], 
                    [self.loss_eval, self.accuracy],
                    givens={
                        self.X_batch: self.dataset['X_valid'][self.batch_slice],
                        self.y_batch: self.dataset['y_valid'][self.batch_slice],
                    },
            )

            self.channels['validation'] = {
                "names":("Validation Loss","Validation Accuracy"),
                "dataset": "Validation",
                "eval": iter_valid,
                "dimensions": ['Loss', 'Accuracy']
                }

    def add_dropout_channels(self):
        """
        Monitor the validation accuracy and loss with dropout.
        """
        # accuracy with dropout
        self.dropout_pred = T.argmax(
            self.output_layer.get_output(self.X_batch, 
                deterministic=False), axis=1)
        self.dropout_accuracy = T.mean(T.eq(self.dropout_pred, self.y_batch), 
                dtype=theano.config.floatX)

        iter_valid = theano.function([self.batch_index], 
                [self.loss_eval, self.accuracy, 
                    self.loss_train, self.dropout_accuracy],
                givens={
                    self.X_batch: self.dataset['X_valid'][self.batch_slice],
                    self.y_batch: self.dataset['y_valid'][self.batch_slice],
                },
        )

        self.channels['validation'] = {
            "names":("Validation Loss","Validation Accuracy",
                "Validation Loss with Dropout", 
                "Validation Accuracy with Dropout"),
            "dataset": "Validation",
            "eval": iter_valid,
            "dimensions": ['Loss', 'Accuracy', 'Loss', 'Accuracy']
        }

    def add_update_ratio_channel(self):
        """
        Crudely add a channel monitoring the global ratio of update norm to 
        parameter norm.
        """
        self.update_ratios = [T.abs_((self.updates[param]-param)/param) 
                for param in self.all_params]
        self.mean_update_ratio = sum(T.mean(p) 
                for p in self.update_ratios)/len(self.update_ratios)
        self.sigma_update_ratio = sum(T.sqrt(T.var(p)) 
                for p in self.update_ratios)/len(self.update_ratios)

        # make channel with the ratio of these (from train channel)
        iter_train = theano.function([self.batch_index], 
                [self.loss_train, self.accuracy, self.mean_update_ratio, self.sigma_update_ratio], 
                updates=self.updates,
                givens={
                    self.X_batch: self.dataset['X_train'][self.batch_slice],
                    self.y_batch: self.dataset['y_train'][self.batch_slice],
                },
        )

        self.channels['train'] = {
            "names":("Train Loss","Train Accuracy", "Mean Update Ratio", "Sigma Update Ratio"),
            "dataset": "Train",
            "eval": iter_train,
            "dimensions": ['Loss', 'Accuracy', 'update/param', 'sigma(update/param)']
            }

def enforce_shared(dataset):
    """
    Datasets as dictionaries containing numpy arrays and those containing
    shared variables will both be returned as dictionaries containing shared
    variables.
    """
    for X_name in [n for n in dataset.keys() if 'X' in n]:
        if not isinstance(dataset[X_name], 
                theano.sandbox.cuda.var.CudaNdarraySharedVariable):
            dataset[X_name] = theano.shared(
                    lasagne.utils.floatX(dataset[X_name]))
    for y_name in [n for n in dataset.keys() if 'y' in n]:
        if not isinstance(dataset[y_name], T.TensorVariable):
            dataset[y_name] = T.cast(dataset[y_name].ravel(), 'int32')
    return dataset

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


