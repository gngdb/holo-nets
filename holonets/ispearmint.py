#!/usr/bin/env python
#
# Some tools to make using spearmint in-notebook slightly less of a hassle.

import spearmint.main
import holoviews as hv
import numpy as np

def make_holomap(experiment_name):
    """
    Given an experiment name will load up the jobs and 
    process them into a HoloMap for easy viewing, slicing
    etc.
    """
    db = spearmint.main.MongoDB()
    jobs = spearmint.main.load_jobs(db, experiment_name)
    N = len(jobs)
    params = jobs[0]['params'].keys()
    # iterate over params building up traces of each
    paramdict = {}
    for p in params:
        paramdict[p] = np.array([j['params'][p]['values']
                                 for j in jobs]).ravel()
    # get trace for the value of interest
    targets = [j.get('values',{'main':1}).values() for j in jobs]
    paramdict['Best Loss'] = np.array(targets).ravel()
    # build into HoloMap
    holomap = hv.HoloMap(key_dimensions=['Channel'])
    for k in paramdict:
        x=zip(range(N), paramdict[k])
        holomap[k] = \
            hv.Curve(zip(range(N), paramdict[k]),
                     key_dimensions=['Iteration'],
                     value_dimensions=[k])
    return holomap

def best(experiment_name):
    """
    Returns where the best value occurred, and what the parameter values were,
    as a dictionary.
    """
    db = spearmint.main.MongoDB()
    jobs = spearmint.main.load_jobs(db, experiment_name)
    sortedjobs = sorted(jobs, key=lambda j: j.get('values',{'main':1.0}).values()[0])
    bestjob = sortedjobs[0]
    best = dict(target=bestjob['values']['main'])
    for param in bestjob['params']:
        best[param] = bestjob['params'][param]['values'][0]
    best['id'] = bestjob['id']
    return best
