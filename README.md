# Overview
This is the documentation repository for the clustering algorithm of the paper "Interpretable Clustering: An Optimization Approach" by Dimitris Bertsimas, Agni Orfanoudaki, and Holly Wiberg. The purpose of this method, ICOT, is to generate interpretable tree-based clustering models. This code is compatible with Julia version 1.1.0, available for download [here](https://julialang.org/downloads/).

# Academic License and Installation
The ICOT software package uses tools from the [Interpretable AI](https://www.interpretable.ai/) suite and thus it requires an academic license. You can request an academic license by emailing <info@interpretable.ai> with your academic institution address and the subject line "Request for ICOT License".

You can download the system image the following links:
* [Linux](https://iai-system-images.s3.amazonaws.com/icot/linux/julia1.1.0/v1.0/sys-linux-julia1.1.0-iai0.1.0-878.zip) 
* [Mac](https://iai-system-images.s3.amazonaws.com/icot/macos/julia1.1.0/v1.0/sys-macos-julia1.1.0-iai0.1.0-878.zip)

You can find detailed installation guidelines for the system image [here](https://docs.interpretable.ai/stable/installation/).

# Algorithm Guidelines

The main command to run the algorithm on a dataset `X` is `ICOT.fit!(learner, X, y);` where the `y` can refer to some data partition that is associated with the dataset. The `learner` is defined as an `ICOT.InterpretableCluster()` object with the following parameters:
* `criterion`
* `ls_warmstart_criterion`
* `kmeans_warmstart`
* `geom_search`
* `geom_threshold`
* `minbucket`
* `max_depth`
* `ls_random_seed`
* `cp`
* `ls_num_tree_restarts`

You can visualize your model on a browser using the `ICOT.showinbrowser()` command.

We have added an example for the ruspini dataset in the `src` folder called `runningICOT_example.jl`.

# Citing ICOT
If you use ICOT in your research, we kindly ask that you reference the original [paper](https://link.springer.com/article/10.1007/s10994-020-05896-2) that first introduced the algorithm:

```
@article{bertsimas2020interpretable,
  title={Interpretable clustering: an optimization approach},
  author={Bertsimas, Dimitris and Orfanoudaki, Agni and Wiberg, Holly},
  journal={Machine Learning},
  pages={1--50},
  year={2020},
  publisher={Springer}
}
```

