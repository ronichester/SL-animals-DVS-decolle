# import numpy as np
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import tonic.transforms as transforms
from decolle_tools import list_sliced_files


#sliced SL-Animals-DVS dataset definition
class AnimalsDvsSliced(Dataset):
    """
    The sliced Animals DVS dataset, customized for DECOLLE format.
    """
    
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 randomCrop, redFactor, binMode):
        """
        #In the DECOLLE paper, they reduce the input from 128x128 to 32x32.
        This is done here by specifying a reduction factor of 0.25, in a way
        that you can choose any factor you want. Keep this factor at 1.0 to
        maintain the original sensor size without Downsampling.
        """
        
        self.slicedDataPath = dataPath + 'sliced_recordings/'   #string  
        self.files = list_sliced_files(np.loadtxt(fileList, dtype='str')) #list [1121 files]
        self.samplingTime = samplingTime                   #1 [ms]
        self.sampleLength = sampleLength                   #500 or 1800 [ms]
        self.nTimeBins = int(sampleLength / samplingTime)  #500 or 1800 bins 
        self.randomCrop = randomCrop                       #boolean
        self.redFactor = redFactor                         #float (0.25)
        self.binMode = binMode                             #string
        #read class file
        self.classes = pd.read_csv(                        #DataFrame
            dataPath + 'SL-Animals-DVS_gestures_definitions.csv')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        #load the sample file (NPY format), class name and index, sensor_shape
        events, class_name, class_index, ss = self.get_sample(index)
        
        #prepare the target vector (one hot encoding)
        target = torch.zeros(19)            #initialize target vector
        target[class_index] = 1             #set target vector
        #now replicate the target for the number of time steps/bins
        target = torch.tile(target[None, :], (self.nTimeBins, 1))
        
        #process the events (using TonicFrames)
        frame_transform = transforms.Compose([
            transforms.Downsample(time_factor=0.001),  #us to ms
            transforms.TimeAlignment(),                #1st event at t=0
            transforms.Downsample(spatial_factor=self.redFactor),
            transforms.ToFrame(                        #events -> frames
                sensor_size = (int(ss[0] * self.redFactor), 
                               int(ss[1] * self.redFactor), 
                               2),
                time_window=self.samplingTime,  #in ms
                ),
            ])
        
        #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
        frames = frame_transform(events)
        
        #crop samples in time
        """
        The 'frames' above have variable length for each sample.
        However, DECOLLE needs a fixed length in order to train in batches.
        This is achieved by cropping the samples into fixed sized crops:
            - 500 ms random crops during training 
            - 1800 ms fixed crops from the sample beginning during testing
        
        Information to be taken into consideration for the SL-Animals dataset:
            - shortest sample: 880 ms. 
            - largest sample: 9466 ms
            - mean sample: 4360 +- 1189 ms stdev.
        """
        if self.randomCrop:  #choose a random crop (training set)
            actual_bins = frames.shape[0]            #actual sample length
            bin_diff = actual_bins - self.nTimeBins  #difference
            min_timebin = 0 if bin_diff <= 0 else np.random.randint(0, bin_diff)
            max_timebin = min_timebin + self.nTimeBins
            frames = frames[min_timebin:max_timebin, ...]
        else:                #get a fixed crop from the start (testing set)
            frames = frames[:self.nTimeBins, ...]
    
        #assure sample has nTimeBins (or pad with zeros)
        if frames.shape[0] < self.nTimeBins:
            padding = np.zeros((self.nTimeBins - frames.shape[0], 
                                2, 
                                int(ss[1] * self.redFactor), 
                                int(ss[0] * self.redFactor)))
            frames = np.concatenate([frames, padding], axis=0)
        
        #input spikes need to be float Tensors shaped TCHW for DECOLLE
        frames = frames.transpose(0,1,3,2)   #TCWH -> TCHW
        input_spikes = torch.Tensor(frames)  #torch.float32

        #choice of binning mode
        """
        By default, Tonic represents the number of spikes at each pixel on every
        time bin as an integer number. Apparently, DECOLLE uses exactly these
        integer numbers in the Tensors - this is the 'SUM' binning mode. Another
        representation used in other SNN methods is the 'OR' binning mode, 
        meaning that if there is either 1 OR more spikes at a specific [x,y] 
        pixel at the same time bin, the pixel value is 1.0 in the Tensor.
        """
        if self.binMode == 'OR' :
            #set all pixels with spikes to the value '1.0'
            input_spikes = torch.where(
                (input_spikes > 0),                 #if spike:
                1.0,                                #set pixel value to 1
                input_spikes)                       #else keep value 0
        elif self.binMode == 'SUM_NORM' :
            #set all pixels with spikes to a normalized SUM value
            input_spikes = torch.where(
                (input_spikes > 0),                 #if spike:
                input_spikes / input_spikes.max(),  #set pixel to range [0, 1.0]
                input_spikes)                       #else keep value 0
        elif self.binMode == 'SUM' :
            #all pixels display the number of spikes (integer) on each time bin
            pass  #do nothing, TonicFrames works natively in 'SUM' mode
        else:
            print("Invalid binning mode; results are compromised!")
            print("(binning_mode should be only 'OR', 'SUM' or 'SUM_NORM')")
        
        return input_spikes, target
    
    def get_sample(self, index):
        #return the sample events, class name, class index and sensor shape
        assert index >= 0 and index <= 1120
   
        #the sample file name
        input_name  = self.files[index]
        
        #load the sample file (NPY format)
        events = np.load(self.slicedDataPath + input_name)
        
        #find sample class
        class_index = index % 19                           #[0-18]
        class_name =  self.classes.iloc[class_index, 1]
        
        sensor_shape = (128, 128)
        
        return events, class_name, class_index, sensor_shape
    
