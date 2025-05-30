import numpy as np
import pandas as pd
from collections import OrderedDict

def upsample_array_nn(original_array, new_array_length, original_sampling_rate, new_sampling_rate):
    """
    Upsample a categorical label array to a higher sampling rate using nearest-neighbor interpolation.
    
    Parameters:
        original_array: 1D array
        new_array_length (int): Desired length of the output array.
        original_sampling_rate (int/float): Original sampling rate (e.g., 100 Hz).
        new_sampling_rate (int/float): Target sampling rate (e.g., 100 Hz).
    
    Returns:
        np.ndarray: Upsampled label array of length `new_array_length`.
    """
    # Calculate scaling factor
    scale_factor = new_sampling_rate / original_sampling_rate
    
    # Upsample by repeating each label proportionally
    upsampled_labels = np.repeat(original_array, int(scale_factor))
    
    # Adjust length to match `new_array_length`
    if len(upsampled_labels) < new_array_length:
        # Pad with the last label
        upsampled_labels = np.append(
            upsampled_labels,
            [upsampled_labels[-1]] * (new_array_length - len(upsampled_labels)))
    elif len(upsampled_labels) > new_array_length:
        # Truncate excess
        upsampled_labels = upsampled_labels[:new_array_length]
    
    return upsampled_labels

def label_extract(all_labels, behavior,interval, sample_rate):
    """
    Function to extract the labels from excel file

    parameters:
        all_labels = pd.read_excel(path+file) (path+file = filepath to excelfile)
        behavior = 'Stiff Movement' (label_extract_legrise) or 'ExtendedLeg' (label_extract)
        interval = np.arange(0,(np.array(array_length)-1)) 
        sample_rate in Hz
    
    # calculation between angles (in degrees) between two vectors in an array
    """
    start_frames = np.round(pd.to_numeric(all_labels['Observation id'].str.split('_',expand=True)[5])/30*sample_rate)
    start_frames_single = np.sort(list(OrderedDict.fromkeys(start_frames)))
    # find start index for behavior
    behavior_start_single = np.round(all_labels['Start (s)'][all_labels['Behavior']==behavior]*sample_rate)
    behavior_start = (behavior_start_single+start_frames[all_labels['Behavior']==behavior]).sort_values()

    # find stop index for behavior
    behavior_stop_single = np.round(all_labels['Stop (s)'][all_labels['Behavior']==behavior]*sample_rate)
    behavior_stop = (behavior_stop_single+start_frames[all_labels['Behavior']==behavior]).sort_values()

    behavior_label_all = np.zeros([interval[-1]+1,1])
    behavior_label = np.zeros([len(interval),1])
    behavior_label_5s = np.zeros([len(start_frames_single),1])
    for st in range(0,len(behavior_start)):
        behavior_label_all[int(behavior_start.iloc[st]):int(behavior_stop.iloc[st])]=1
        # classify as 1 if 15 frames(0.5s) are labeled as 1
    nrsf = 0
    for sf in start_frames_single:
        if sum(behavior_label_all[int(sf):int(sf+(5*sample_rate))])>=sample_rate/2:
            behavior_label_5s[nrsf]=1
        nrsf=nrsf+1
    
    behavior_label = np.take(behavior_label_all, interval)
    
    return behavior_label, behavior_label_5s, start_frames_single

def label_extract_legrise(all_labels, behavior,interval,start_frames_single, sample_rate):
    # find behavior start and stop index
    behavior_times = np.round(all_labels['Image index'][all_labels['Behavior']==behavior])/30*sample_rate
    # find start index for behavior
    behavior_start = behavior_times[0:-1:2]

    # find stop index for behavior
    behavior_stop = behavior_times[1:len(behavior_times)+1:2]

    behavior_label_all = np.zeros([interval[-1]+1,1])
    behavior_label = np.zeros([len(interval),1])
    behavior_label_5s = np.zeros([len(start_frames_single),1])
    
    for st in range(0,len(behavior_start)):
        behavior_label_all[int(behavior_start.iloc[st]):int(behavior_stop.iloc[st])]=1
    
    # classify as 1 if 15 frames(0.5s) are labeled as 1
    nrsf = 0
    # for sf in range(0,interval[-1],5*100):
    #     if sum(behavior_label_all[sf:sf+5*100])>=100/2:
    #         behavior_label_5s[nrsf]=1
    #     nrsf=nrsf+1
    for sf in start_frames_single:
        if sum(behavior_label_all[int(sf):int(sf+(5*sample_rate))])>=sample_rate/2:
            behavior_label_5s[nrsf]=1
        nrsf=nrsf+1
    
    behavior_label = np.take(behavior_label_all, interval)
    
    return behavior_label,behavior_label_5s
