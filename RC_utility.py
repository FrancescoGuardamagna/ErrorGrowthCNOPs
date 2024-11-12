import scipy
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from utility import compute_NRMSE
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import pickle
from collections import defaultdict
from utility import compute_monthly_data
from sklearn.metrics import mean_squared_error


class RC:

    def __init__(self,Nx,alpha,connectivity,
    sp,input_scaling,reguralization,
    input_variables,normalize=True):

        self.variables_number=input_variables

        self.Nx=Nx
        self.input_scaling=input_scaling

        reservoir_distribution=scipy.stats.uniform(loc=-1,scale=2).rvs
        W=scipy.sparse.random(Nx,Nx,density=connectivity,data_rvs=reservoir_distribution)
        W=W.toarray()
        eigs,_=np.linalg.eig(W)
        max_eig=np.max(np.abs(eigs))
        W=W/max_eig
        W=W*sp
        self.W=W

        self.Win=np.random.uniform(low=-self.input_scaling,
        high=self.input_scaling,size=(self.Nx,input_variables))

        self.trainable_model=Ridge(alpha=reguralization,fit_intercept=True)
        self.alpha=alpha

        self.annual_period=12
        self.normalize=normalize


    def esn_transformation(self,X):

        time_steps=X.shape[1]
        previous_state=np.zeros(shape=(self.Nx,1))

        for i in range(time_steps):

            current_input=X[:,i]
            current_input=current_input[:,np.newaxis]

            update=np.tanh(np.matmul(self.Win,current_input)+np.matmul(self.W,previous_state))
            next_state=(1-self.alpha)*previous_state+self.alpha*update

            previous_state=next_state

            if(i==0):
                esn_representation=next_state.T
            else:
                esn_representation=np.concatenate((esn_representation,next_state.T),axis=0)
        
        return esn_representation

    def train(self,X,spin,lead_time,cycle):

        reservoir_representation=self.esn_transformation(X)
        
        if(np.isnan(reservoir_representation).any()):
            print("Reservoir representation contains Nan not possible to train")
            return 

        reservoir_representation=reservoir_representation[spin:-lead_time,:]

        Y=X.T
        if(cycle):
            Y=Y[spin+lead_time:,[i for i in range(1,Y.shape[1])]]
        else:
            Y=Y[spin+lead_time:,:]
        self.trainable_model.fit(reservoir_representation,Y)
        self.W_out=self.trainable_model.coef_
        self.bias_redout=self.trainable_model.intercept_
    
    def predict(self,X,spin,lead_time):

        reservoir_representation=self.esn_transformation(X)
        reservoir_representation=reservoir_representation[spin:-lead_time,:]
        predictions=self.trainable_model.predict(reservoir_representation)

        return predictions

    def integrator(self,initial_conditions,iterations,r0,
    cycle,time_line_validation):

        previous_state=r0
        previous_state=np.reshape(r0,(r0.shape[0],1))

        initial_conditions=np.array(initial_conditions)
        time_series=initial_conditions[np.newaxis,:]
        initial_conditions=initial_conditions[:,np.newaxis]

        current_input=initial_conditions

        for i in range(iterations):

            update=np.tanh(np.matmul(self.Win,current_input)+np.matmul(self.W,previous_state))
            next_state=(1-self.alpha)*previous_state+self.alpha*update
            output=self.trainable_model.predict(next_state.T)

            if(cycle):
                initial_time=time_line_validation[i+1]
                output=np.insert(output,0,np.sin((2*np.pi*initial_time)/self.annual_period))
                output=output[np.newaxis,:]

            time_series=np.concatenate((time_series,output),axis=0)
            
            previous_state=next_state
            current_input=output.T
        
        return time_series

    def return_reconstruction(self,Data,spin,
    cycle,time_line_validation,total_lenght,
    apply_perturbation=False,perturbation=None):

        if(spin!=0):

            spin_data=self.esn_transformation(Data[:,:spin])
            r0=spin_data[-1,:]
            initial_conditions=Data[:,spin].copy()
            time_line_validation=time_line_validation[spin:].copy()
            iterations=total_lenght

            if(apply_perturbation):
                initial_conditions[1]=initial_conditions[1]+perturbation[0]
                initial_conditions[2]=initial_conditions[2]+perturbation[1]
                initial_conditions[3]=initial_conditions[3]+perturbation[2]
                
            Reconstructed_Data=self.integrator(initial_conditions,
            iterations,r0,cycle=cycle,time_line_validation=time_line_validation)


        else:

            r0=np.zeros(self.Nx)
            initial_conditions=Data[:,spin]

            if(apply_perturbation):
                initial_conditions[1]=initial_conditions[1]+perturbation[0]
                initial_conditions[2]=initial_conditions[2]+perturbation[1]
                initial_conditions[3]=initial_conditions[3]+perturbation[2]

            iterations=total_lenght
            Reconstructed_Data=self.integrator(initial_conditions,iterations,r0,
            cycle=cycle,time_line_validation=time_line_validation)

        return Reconstructed_Data
    
    def return_full_time_series(self,Data,spin,
    cycle,time_line_validation,lead,apply_perturbation=False,
    perturbation=None,compute_monthly=False):

        total_time_series=[]

        for i in range(0,Data.shape[1]-(spin+lead)):

            prediction_tmp=self.return_reconstruction(Data[:,i:],spin,
            cycle,time_line_validation[i:],lead,apply_perturbation,
            perturbation)

            total_time_series.append(prediction_tmp[-1,:])
        
        y_validation=Data[:,spin+lead:].T
        total_time_series=np.array(total_time_series)
        time_line_validation=time_line_validation[spin+lead:]

        if(self.normalize):
            total_time_series[:,1]=total_time_series[:,1]*2
            y_validation[:,1]=y_validation[:,1]*2
            total_time_series[:,2]=total_time_series[:,2]*50
            y_validation[:,2]=y_validation[:,2]*50
            total_time_series[:,3]=total_time_series[:,3]*50
            y_validation[:,3]=y_validation[:,3]*50

        if(compute_monthly):
            y_validation=compute_monthly_data(y_validation,time_line_validation)
            total_time_series=compute_monthly_data(total_time_series,time_line_validation)
        
        TE_reconstructed=total_time_series[:,1]
        TE_validation=y_validation[:,1]
        HE_reconstructed=total_time_series[:,2]
        HE_validation=y_validation[:,2]
        HW_reconstructed=total_time_series[:,3]
        HW_validation=y_validation[:,3]

        corr_TE=pearsonr(TE_reconstructed,TE_validation)[0]
        corr_HE=pearsonr(HE_reconstructed,HE_validation)[0]
        corr_HW=pearsonr(HW_reconstructed,HW_validation)[0]
        rms_TE=np.sqrt(np.mean((TE_reconstructed-TE_validation)**2))
        rms_HE=np.sqrt(np.mean((HE_reconstructed-HE_validation)**2))
        rms_HW=np.sqrt(np.mean((HW_reconstructed-HW_validation)**2))

        rms_total=mean_squared_error(y_validation,total_time_series,multioutput="raw_values")
        range_test = np.std(y_validation, axis=0)
        rms_total=rms_total/range_test
        rms_total=np.mean(rms_total[1:])

        return total_time_series,corr_TE,corr_HE,corr_HW,rms_TE,rms_HE,rms_HW,rms_total






