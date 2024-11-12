import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from collections import defaultdict

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def compute_NRMSE(real,predicted):

    rmse = np.sqrt(mean_squared_error(real, predicted))
    value_range = np.std(real)
    nrmse = rmse / value_range
    return nrmse

def compute_monthly_data(Data,time_line):

    time_line_monthly=np.int_(time_line)

    if(len(Data.shape)>1):
        num_variables = Data.shape[1]
    else:
        num_variables=1

  
    counter=0
    initial_index=time_line[0]
    current_monthly_value=np.zeros((1,num_variables))
    time_line_monthly=[]

    for value, index in zip(Data,time_line):
        
        current_monthly_value+=value
        counter+=1
        if(int(initial_index)!=int(index)):
            if(initial_index==time_line[0]):
                monthly_array=current_monthly_value/counter
            else:
                monthly_array=np.concatenate((monthly_array,current_monthly_value/counter),axis=0)
            counter=0
            current_monthly_value=np.zeros((1,num_variables))
            initial_index=index
            time_line_monthly.append(int(index))
    
    monthly_array=np.squeeze(monthly_array)

    return monthly_array




