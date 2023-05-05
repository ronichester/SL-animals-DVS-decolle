# SL-animals-DVS training with DECOLLE
This repository contains a custom DECOLLE (Deep Continuous Local Learning) implementation on the SL-Animals-DVS dataset using Pytorch and the DECOLLE package software.

**A BRIEF INTRODUCTION:**  
DECOLLE is an online training method that directly trains a Spiking Neural Network (SNN), updating the network weights at each single time step. 
Therefore, it is a suitable method for online training SNNs, which are biologically plausible networks (in short).  
The SL-animals-DVS is a dataset of sign language (SL) gestures peformed by different people representing animals, and recorded with a Dynamic Vision Sensor (DVS).  

<p align="center">
<img src="https://github.com/ronichester/SL-animals-DVS-slayer/blob/main/samples_and_outputs/SL_animals_sample2.gif" width="600px"></p>

<p align="center">
<img src="https://github.com/ronichester/SL-animals-DVS-slayer/blob/main/samples_and_outputs/SL_animals_sample10.gif" width="600px"></p>

<p align="center"> </p>  

The reported results in the SL-animals paper were divided in two: results with the full dataset and results with a reduced dataset, meaning excluding group S3. The results achieved with the implementation published here fall short of the published results, but get fairly close, considering the published results have no code available to reproduce them.  
  
**The implementation published in this repository is the first publicly available DECOLLE implementation on the SL-animals dataset** (and the only one as of may 2023, as far as I know). The results are summarized below:

|       | Full Dataset | Reduced Dataset |
|:-:|:-:|:-:|
| Reported Results    | 70.6 +- 7.8 % | 77.6 +- 6.5 % |
| This Implementation | 62.19 +- 3.99 % | 62.91 +- 4.04 % |

           
## Requirements
While not sure if the list below contains the actual minimums, it will run for sure if you do have the following:
- Python 3.0+
- Pytorch 1.11+
- CUDA 11.3+
- decolle (installation instructions [here](https://github.com/nmi-lab/decolle-public))
- python libraries: os, numpy, matplotlib, pandas, sklearn, datetime, tonic, pyyaml, h5py, tensorboardX

## README FIRST
This package contains the necessary python files to train a Spiking Neural Network with a custom DECOLLE method (a slightly modified version) on the Sign Language Animals DVS dataset. 

**IMPLEMENTATION**  
Package Contents:  
- dataset.py
- decole_tools.py
- train_test_only.py
- parameters/params_slanimals.yml
- decolle1 (folder with the custom decolle method)

The SL-Animals-DVS dataset implementation code is in *dataset.py*, and it's basically a Pytorch Dataset object. The library [*Tonic*](https://tonic.readthedocs.io/en/latest/index.html#) was used to read and process the DVS recordings.  
Some auxiliary functions to slice the dataset, split the dataset, plot and animate dataset samples and some getters are in *decolle_tools.py*.  
The main program is in *train_test_only.py*, which uses a simple, yet not optimal, experimental procedure for training a network using cross validation after dividing the dataset into train and test sets. This was done in an effort to replicate the published results.  
The file *params_slanimals.yml* contains the main parameters that can be customized like *batch size*, *sampling time*, *sample length*, *neuron type*, *data path*, and many others.  
Finally, the *decolle1* folder contains the original decolle software implementation, slightly modified for outputting plots, easier display while training, and minor changes.
 
## Use
1. Clone this repository:
```
git clone https://github.com/ronichester/SL-animals-DVS-decolle
```
2. Download the dataset in [this link](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database);
3. Save the DVS recordings in the *data/recordings* folder and the file tags in the *data/tags* folder;
4. Edit the custom parameters according to your preferences in *parameters/params_slanimals.yml*. The default parameters setting is functional and was tailored according to the information provided in the relevant papers, the reference codes used as a basis, and mostly by trial and error (lots of it!). You are encouraged to edit the main parameters and **let me know if you got better results**.
6. Run *train_test_only.py* to start the SNN training:
```
python train_test_only.py
```
7. The network weights, training curves and tensorboard logs will be saved in *src/results*. To visualize the training with Tensorboard:
  - open a terminal (I use Anaconda Prompt), go to the *src* directory and type:
```
tensorboard --logdir=results
```
  - open your browser and type in the address bar http://localhost:6006/ or any other address shown in the terminal screen.
  

## References 
- Vasudevan, A., Negri, P., Di Ielsi, C. et al. ["SL-Animals-DVS: event-driven sign language animals dataset"](https://doi.org/10.1007/s10044-021-01011-w) . *Pattern Analysis and Applications 25, 505–520 (2021)*. 
- Kaiser, Jacques and Mostafa, Hesham and Neftci, Emre; [Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)](https://www.frontiersin.org/article/10.3389/fnins.2020.00424); *Frontiers in Neuroscience, vol.14 pages 424 (2020)*
- The original dataset can be downloaded [here](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database)

## Copyright
Copyright 2023 Schechter Roni. This software is free to use, copy, modify and distribute for personal, academic, or research use. Its terms are described under the General Public License, GNU v3.0.
