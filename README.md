[![Stories in Ready](https://badge.waffle.io/gngdb/holo-nets.png?label=ready&title=Ready)](https://waffle.io/gngdb/holo-nets)

![Travis Badge](https://travis-ci.org/gngdb/holo-nets.svg?branch=master)

# holo-nets

Simple wrapper around Holoviews for training neural networks interactively and monitoring channels.

Currently in development.

Goals
-----

This should be a modular wrapper for Holoviews that allows you to pass a 
training function that will return channel values in a dictionary on each epoch
and accumulate these into curves for plotting. Then, these will be accessed as 
a [Holomap][] using [Holoviews][] for easy interactive viewing in an IPython 
notebook. 

Along with this, a simple training function will also be provided that is well 
suited for small datasets that you might want to run interactively. It will 
assume the data will be accessible as a shared variable in Theano, and iterate
over it by slicing.

Additional features for development:

* Asynchronous evaluation - starting jobs which save to pickle files and not
blocking the IPython notebook.
* Choosing GPU - select which GPU to use when starting a job.
* Asynchronous grid search - accumulate jobs to run asynchronously and set them
off automatically on different GPUs, saving to the same pickle file.

[holomap]: https://ioam.github.io/holoviews/Tutorials/Exploring_Data
[holoviews]: https://ioam.github.io/holoviews/index.html

