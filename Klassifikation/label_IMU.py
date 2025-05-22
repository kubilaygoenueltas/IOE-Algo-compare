### Function to extract the labels from excel file

# all_labels = pd.read_excel(path+file) (path+file = filepath to excelfile)
# behavior = 'Stiff Movement' (label_extract_legrise) or 'ExtendedLeg' (label_extract)
# interval = np.arange(0,(np.array(array_length)-1)) 

# calculation between angles (in degrees) between two vectors in an array
import numpy as np
import pandas as pd
from collections import OrderedDict

def label_extract(all_labels, behavior,interval):

    start_frames = np.round(pd.to_numeric(all_labels['Observation id'].str.split('_',expand=True)[5])/30*52)
    start_frames_single = np.sort(list(OrderedDict.fromkeys(start_frames)))
    # find start index for behavior
    behavior_start_single = np.round(all_labels['Start (s)'][all_labels['Behavior']==behavior]*52)
    behavior_start = (behavior_start_single+start_frames[all_labels['Behavior']==behavior]).sort_values()

    # find stop index for behavior
    behavior_stop_single = np.round(all_labels['Stop (s)'][all_labels['Behavior']==behavior]*52)
    behavior_stop = (behavior_stop_single+start_frames[all_labels['Behavior']==behavior]).sort_values()

    behavior_label_all = np.zeros([interval[-1]+1,1])
    behavior_label = np.zeros([len(interval),1])
    behavior_label_5s = np.zeros([len(start_frames_single),1])
    for st in range(0,len(behavior_start)):
        behavior_label_all[int(behavior_start.iloc[st]):int(behavior_stop.iloc[st])]=1
        # classify as 1 if 15 frames(0.5s) are labeled as 1
    nrsf = 0
    for sf in start_frames_single:
        if sum(behavior_label_all[int(sf):int(sf+(5*52))])>=52/2:
            behavior_label_5s[nrsf]=1
        nrsf=nrsf+1
    
    behavior_label = np.take(behavior_label_all, interval)
    
    return behavior_label, behavior_label_5s, start_frames_single

def label_extract_legrise(all_labels, behavior,interval,start_frames_single):
    # find behavior start and stop index
    behavior_times = np.round(all_labels['Image index'][all_labels['Behavior']==behavior])/30*52
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
    # for sf in range(0,interval[-1],5*52):
    #     if sum(behavior_label_all[sf:sf+5*52])>=52/2:
    #         behavior_label_5s[nrsf]=1
    #     nrsf=nrsf+1
    for sf in start_frames_single:
        if sum(behavior_label_all[int(sf):int(sf+(5*52))])>=52/2:
            behavior_label_5s[nrsf]=1
        nrsf=nrsf+1
    
    behavior_label = np.take(behavior_label_all, interval)
    
    return behavior_label,behavior_label_5s

