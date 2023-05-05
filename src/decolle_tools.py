# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:39:43 2022

@author: Schechter
"""
#import librarie
import os
import yaml
import tonic
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tonic.transforms as transforms
from sklearn.model_selection import KFold
from collections import defaultdict 

#slice the dataset
def slice_data(data_path, file_list):
    """
    A script to slice the SL animals DVS recordings into actual samples for 
    training, and save the slices to disk. 
    
    Parameters:
        data_path: str;
            The 'raw' data path, for the 59 dvs recordings
        file_list: txt file;
            A text file with a list of the 'raw' file names
    """
    print('Checking for sliced dataset:')
    
    #create sliced dataset directory and path
    os.makedirs(data_path + "sliced_recordings", exist_ok=True)
    sliced_data_path = data_path + "sliced_recordings/"
    
    #load file names into a 1D array
    files = np.loadtxt(file_list, dtype='str')  #1D array [max 59]
    
    #check if dataset is already sliced
    if len(os.listdir(sliced_data_path)) < (19 * len(files)):
        
        print('Slicing the dataset, this may take a while...')
        
        #For each of the raw recordings: slice in 19 pieces and save to disk
        for record_name in files:
            print('Processing record {}...'.format(record_name))
            
            #read the DVS file
            """
            The output of this function:
                sensor_shape: tuple, the DVS resolution (128, 128)
                events: 1D-array, the sequential events on the file
                        1 microsecond resolution
                        each event is 4D and has the shape 'xytp'
            """
            sensor_shape, events = tonic.io.read_dvs_128(data_path + 'recordings/'
                                                         + record_name + '.aedat')
            #read the tag file
            tagfile = pd.read_csv(data_path + 'tags/' + record_name + '.csv')  #df
            
            #define event boundaries for each class
            events_start = list(tagfile["startTime_ev"])
            events_end = list(tagfile["endTime_ev"])
            
            #create a list of arrays, separating the recording in 19 slices
            sliced_events = tonic.slicers.slice_events_at_indices(events, 
                                                                 events_start, 
                                                                 events_end)
            #save 19 separate events on disk
            for i, chosen_slice in enumerate(sliced_events):
                np.save(sliced_data_path + '{}_{}.npy'.format(
                    record_name, str(i).zfill(2)), chosen_slice)
        print('Slicing completed.\n')
        
    else:
        print('Dataset is already sliced.\n')


#visualize dataset sample on animation
def animate_events(dataset, sample_index, time_window=50, frame_rate=24, repeat=False):
    """
    Generates an animation on a dataset sample. The sample is retrieved
    as a 1D array of events in the Tonic format (x, y, t, p), in [us] 
    resolution.
    
        dataset: torch Dataset object
        sample_index: int, must be between [0, 1120]
        time_window: int, time window in ms for each frame (default 50 ms)
        frame_rate: int, (default 24 FPS)
        repeat: bool, loop the animation infinetely (default is False)
    """
    #get sample events, class name, class index (ci) and sensor shape (ss)
    sample_events, sample_class, ci, ss = dataset.get_sample(sample_index)
       
    #create transform object
    sensor_size = (ss[0], ss[1], 2)                     #DVS sensor size
    frame_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),       # us to ms
        transforms.TimeAlignment(),                     # 1st event at t=0
        transforms.ToFrame(sensor_size,                 # bin into frames
                           time_window=time_window)     # in [ms]
        ])  
    
    #transform event array -> frames (shape TCWH (time_bins, 2, 128, 128))
    frames = frame_transform(sample_events)
    
    #interval between frames in ms (default=41.6)
    interval = 1e3 / frame_rate  # in ms
    
    #defining 1st frame: image is the difference between polarities
    fig = plt.figure()
    plt.title("class name: {}".format(sample_class))
    image = plt.imshow((frames[0][1]-frames[0][0]).T, cmap='gray')
    
    #update the data on each frame
    def animate(frame):
        image.set_data((frame[1]-frame[0]).T)  
        return image

    animation = anim.FuncAnimation(fig, animate, frames=frames, 
                                   interval=interval, repeat=repeat)
    plt.show()
    
    return animation


#visualize dataset sample on plots (by time bins)
def plot_events(dataset, sample_index):
    """
    Generates a plot with 3 frames on a dataset sample. The events of a sample
    are divided in 3 time bins, each frame accumulates the events of 1 bin.
    """
    #get sample events, class name, class index (ci) and sensor shape (ss)
    sample_events, sample_class, ci, ss = dataset.get_sample(sample_index)
    
    #transform event array -> frames (shape TCWH)
    sensor_size = (ss[0], ss[1], 2)                     #DVS sensor size
    frame_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),       # us to ms
        transforms.TimeAlignment(),                     # 1st event at t=0
        transforms.ToFrame(sensor_size, n_time_bins=3)  # events -> 3 frames
        ])  
    frames = frame_transform(sample_events)

    def plot_frames(frames):
        fig, axes = plt.subplots(1, len(frames))
        fig.suptitle("class name: {}".format(sample_class))
        for axis, frame in zip(axes, frames):
            axis.imshow((frame[1] - frame[0]).T, cmap='gray')
            axis.axis("off")
        plt.tight_layout()
        plt.show()
    
    plot_frames(frames)
    
    return frames


def kfold_split(fileList, seed, export_txt=False):
    """
    Split a file list (txt file) in 4 folds for cross validation (75%, 25%).
    It shuffles the files and then returns 4 separate training and test lists. 
    Optionally export the lists as txt files (default=False).
    
    Returns a generator.
    """
    def gen():  
        #load the files from .txt to an numpy 1D array
        files = np.loadtxt(fileList, dtype='str')  #[max 59 recordings]
        #create KFold object
        kf = KFold(n_splits=4, shuffle=True, random_state=seed)
        #create the folds
        for i, (train_index, test_index) in enumerate(kf.split(files), start=1):
            train_set, test_set = files[train_index], files[test_index]
            if export_txt:
                np.savetxt('../data/trainlist{}.txt'.format(i), train_set, 
                           fmt='%s')
                np.savetxt('../data/testlist{}.txt'.format(i), test_set, 
                           fmt='%s')
            yield train_set, test_set
    return gen()  #returns a generator!


def list_sliced_files(raw_file_list):
    #create a list of sliced files, given a list of 'raw' recording files
    sliced_file_list = []
    for file in raw_file_list:
        for i in range(19):
            sliced_file_list.append(file + '_{}.npy'.format(str(i).zfill(2)))
    
    return sliced_file_list


# Dacay learning_rate
def adjust_learning_rate(optimizer, epoch, factor, lr_decay_epoch=25):
    """Decay learning rate by a factor every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor
    print('Optimizer Learning Rate:', optimizer.param_groups[0]['lr'])
    return optimizer


def get_params(yaml_file):
    """
    Get configuration parameters stored in a 'yaml' file.
    """
    with open(yaml_file, 'r') as f: 
        params = yaml.safe_load(f) 
        return defaultdict(lambda: None, params)


def get_optimizer(net, params):
    if params['optimizer'] == 'adam':
        opt = optim.Adam(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    elif params['optimizer'] == 'adamax':
        opt = optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    elif params['optimizer'] == 'adagrad':
        opt = optim.Adagrad(net.get_trainable_parameters(), lr=params['learning_rate'])
    else:
        raise ValueError('Please provide a valid entry for optimizer. Possible choices are: adam, adamax, adagrad')
    return opt


def get_loss(loss):
    if loss == 'mse':
        return nn.MSELoss()
    elif loss == 'smoothL1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError('Please provide a valid entry for loss. Possible choices are: mse, smoothL1')
