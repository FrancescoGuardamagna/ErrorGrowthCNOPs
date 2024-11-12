from RC_utility import RC
import optuna
import numpy as np
import matplotlib.pyplot as plt
import sys
from utility import compute_NRMSE
import math
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
from ZebiakCaneDataUtility import ZebiakCaneModel
import os

def ObjectiveZC(trial,
weekly,spin,transitional_years,
train_years,validation_years,variables,
drag):

    try:

        Nx=trial.suggest_int('Nx',50,600)
        alpha=trial.suggest_float('alpha',0.1,1.0)
        connectivity=trial.suggest_float('connectivity',0.01,0.2)
        sp=trial.suggest_float('sp',0.6,1.5)
        input_scaling=trial.suggest_float('input_scaling',0.01,10.0)
        reguralization=trial.suggest_float('reguralization',0.000000000001,0.1)

        #################################
        #Define RC and ZC
        input_variables_number=len(variables)+1
        zc=ZebiakCaneModel("../ZebiakCaneFiles/NoisyTrajectories_for_experiments_paper",drag,transitional_years,
        train_years,validation_years,variables,weekly,normalize=True)
        lead_time=18

        if(weekly):
            lead_time=(lead_time)*3
        #################################

        X_train,X_validation=zc.X_train,zc.X_validation
        time_line_validation=zc.time_line_validation
        attemps=5

        rmse_final=[]

        for i in range(attemps):

            rc=RC(Nx,alpha,connectivity,sp,
            input_scaling,reguralization,input_variables_number,
            weekly,normalize=True)
            rc.train(X_train,spin,1,cycle=True)

            print("iteration:{}".format(i))
            print("drag:{}".format(drag))

            print("lead time:{}".format(lead_time))
            sys.stdout.flush()
            predictions,corr_TE,corr_HE,corr_HW,_,_,_,rms_total=rc.return_full_time_series(X_validation.copy(),spin,
            True,time_line_validation,lead_time,False,
            None)

            predictions=predictions[np.newaxis,:,1:input_variables_number]

            if(i==0):
                predictions_set=predictions
            else:
                predictions_set=np.concatenate((predictions_set,predictions),axis=0)

            print("############################################")
            print("\n")
            print("Total RMS:{}".format(rms_total))
            print("correlation TE:{}".format(corr_TE))
            print("correlation HE:{}".format(corr_HE))
            print("correlation HW:{}".format(corr_HW))
            print("\n")
            print("#############################################")
            sys.stdout.flush()
            rmse_final.append(rms_total)
            del rc
        return np.mean(rmse_final)

    except Exception as e:

        return sys.float_info.max

def print_status(study,trial):
    print("Trial {} completed with value: {}\n".format(trial.number,trial.value))
    print("Best trial so far: {} with value: {}\n".format(study.best_trial.number,study.best_value))
    print("Current best parameters: {}\n\n".format(study.best_params))
    sys.stdout.flush()



def optuna_optimization(objective_function,direction,n_trials,spin,weekly,
transitional_years,train_years,validation_years,drag,variables,use_spectral,seed):

    study=optuna.create_study(directions=direction)
        
    if(objective_function=="Zebiak"):
        study.optimize(lambda x: ObjectiveZC(x,spin=spin,weekly=weekly,
        train_years=train_years,transitional_years=transitional_years,
        validation_years=validation_years,variables=variables,
        drag=drag,use_spectral=use_spectral,seed=seed),callbacks=[print_status], n_trials=n_trials)
        best_trials=study.best_trials
        sys.stdout.flush()
    return best_trials

spin=180
transitional_years=100
train_years=300
validation_years=[400,600]
weekly=True

drag=0.77
variables=["Nino3","ThE","ThW","Wind"]

print("code starting")
print("###################################################")
sys.stdout.flush()
best_trials=optuna_optimization("Zebiak",
direction=['minimize'],n_trials=50,spin=spin,
weekly=weekly,transitional_years=transitional_years,
train_years=train_years,validation_years=validation_years,
drag=drag,variables=variables)
print(best_trials)