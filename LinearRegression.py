import scipy
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from utility import compute_NRMSE
from scipy.signal import find_peaks
from scipy.stats import pearsonr


class LinearRegressor:

    def __init__(self):

        self.linear_model=LinearRegression()

    def train(self,X_train,lead_time):

        X_train_tmp=X_train[:,:-int(lead_time)]
        labels=X_train[1,int(lead_time):]
        X_train_tmp=X_train_tmp.T
        self.linear_model.fit(X_train_tmp,labels)
    
    def predict(self,X_validation,spin,lead_time):

        X_validation_tmp=X_validation[:,spin:-lead_time]
        X_validation_tmp=X_validation_tmp.T
        prediction=self.linear_model.predict(X_validation_tmp)
        
        return prediction









