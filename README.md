# Universal Representation of Chemical Entities for Machine Learning

## Abstract

Molecular and material machine learning have seen tremendous progress in the past few years, especially thanks to the usage of GNNs. Data representation of chemical
entities is a key component to the success of ML in these fields. Nonetheless, almost of the work done is focused on model architecture or a data representation
tightly coupled with some specific architecture design (SE-3 transformers, ...). Thus, to the best of our knowledge, we have developed the first universal and model-
agnostic graph representation of chemical entities. In this work, we show how this representation allows us to get good performance on a wide variety of molecular
and crystal datasets, discuss the different components of the data representation by how they affect downstream performance, and the possibility of transfer learning.
This new data representation paves the way for a possible crossover of knowledge between molecular and matieral science, whether through transfer learning or joint effort in modelling/data collection spanning multiple chemical fields.

## Experiments

To run the experiments, do as follows:

1. Install the environment using `conda env create -f environment.yaml --name ucr`.

2. Run the results script, './results.sh'


