# SL-animals-DVS training with DECOLLE
This repository contains a DECOLLE (Deep Continuous Local Learning) implementation on the SL-Animals-DVS dataset using Pytorch and the DECOLLE package software.

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
