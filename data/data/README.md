The molecule geometries as well as the silicon geometries in the various training and test sets and the results of the corresponding DFT reference calculations are available.
The data sets were used to generate the results in the paper. The examples in `tbmalt/examples` show how to use the data sets to realize the machine learning training and reproduce the results in the paper.

# Description of dataset.h5

This data set includes geometries of [ANI-1 data set](https://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a) and DFT calculations by using FHI-aims.

`dataset.h5` includes three training runs and testing runs. Each training run and testing run include 1000 and 400 molecules, respectively. We have trained each training run data set and use the corresponding testing run to test the performance of training model. The results in section IV.A.1 are the averages over the test sets of three independent training runs.

# Description of dataset_dos.h5 and siband.hdf5

The usage of `dataset_dos.h5` and `siband.hdf5` can reproduce the results shown in the Section IV.A.2 of the main text.

In detail, dataset_dos.h5 consists of the geometries of randomly rattled silicon systems as well as the DFT references for different training and testing runs.
`dataset_dos.h5` includes four groups named `run1`, `run2`, `run3` and `run_transfer`.
`run1`, `run2` and `run3`, which contribute to the general tests shown in Fig.3 (b) and (c), consist of 30 systems for training and 20 systems for testing, respectively.
`run_transfer` is used for the transferability test shown in Fig.3 (d). It has a 64-atom system in the training set and a 512-atom system in the testing set.

`siband.hdf5` contains the siband-1-1 Slater-Koster parameter set for the Si-Si pair, with additional five grid points contributing to a smooth tail.