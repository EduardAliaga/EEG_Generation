import pandas as pd
import numpy as np

def read_subject_data(data_path, sensors):
    inquiries = []
    for sensor in sensors:
        inquiry = pd.read_csv(f'{data_path}/data_{sensor}.csv')
        inquiry = np.array(inquiry)
        inquiries.append(np.transpose(inquiry))
    timings = np.array(pd.read_csv(f'{data_path}/timing.csv'))
    labels = np.array(pd.read_csv(f'{data_path}/labels.csv'))
    return inquiries, timings, labels

def compute_stimuli(timings, labels):
    sensors_stimulis = []
    for sample in range(0,len(timings)):
        stimuli = np.zeros(396)
        for time_step in range(0, len(timings[0])):
            start = timings[sample][time_step]
            end = start + 10
            if labels[sample][time_step] == 1:
                stimuli[start:end] = 1
            else:
                stimuli[start:end] = 0.5
        sensors_stimulis.append([stimuli, stimuli, stimuli])
    return sensors_stimulis