#!/usr/bin/env python
#
# Some tools to make using spearmint in-notebook slightly less of a hassle.

import spearmint.main
import holoviews as hv
import numpy as np
import itertools

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

def scatter_layout(traces, heat="Best Loss"):
    """
    Takes a holomap of traces and turns it into a set of scatter heatmaps 
    between each pair of parameters. Cell must be run with:

    %%opts Points [color_index=2]
    """
    heatmaps = []
    params = [k for k in traces.keys() if k != heat]
    for p in itertools.combinations(params, 2):
        d = np.array([traces[p[0]].data[:,1],
                      traces[p[1]].data[:,1],
                      traces[heat].data[:,1]]).T
        heatmaps.append(
                hv.Points(d, key_dimensions=[p[0], p[1]], 
                    value_dimensions=[heat])
                )
    return hv.Layout(heatmaps)

def bars(traces, target='Best Loss'):
    hm = hv.HoloMap(key_dimensions=[target])
    params = [k for k in traces.keys() if k != target]   
    sample_arrays = []
    for p in params:
        # turn holomap into arrays:
        sample_arrays.append(traces[p].data[:,1])
    data = np.array(sample_arrays).T
    for r,t in zip(data, traces[target].data[:,1]):
        bardata = [(p,v) for v,p in zip(r,params)]
        hm[t] = hv.Bars(bardata, key_dimensions=['Parameters'], 
                value_dimensions=['Values'])
    return hm
