import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
from utility import compute_NRMSE
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from RC_utility import RC
import pandas as pd
import seaborn as sns
from matplotlib import rc, rcParams
from ZebiakCaneDataUtility import ZebiakCaneModel
import json
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from utility import compute_monthly_data
import sys
import pickle
import argparse
from LinearRegression import LinearRegressor
import best_parameters_validation
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 5
rcParams['xtick.major.size'] = 35
rcParams['ytick.major.width'] = 5
rcParams['ytick.major.size'] = 35
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"
rcParams['font.size']=35

def plot_performances(drag,transitional_years,
train_years,
validation_years,spin=180,
lead_times=[3,6,9,12,15],
variables=['Nino3','ThE','ThW','Wind'],
attemps=50):

    zc=ZebiakCaneModel("../ZebiakCaneFiles/NoisyTrajectories_for_experiments_paper",drag,
    transitional_years,train_years,validation_interval=validation_years,
    variables=variables,normalize=True)

    X=zc.data

    X_validation=zc.X_validation

    correlation_list_RC_4v=[]
    correlation_list_LR_4v=[]
    correlation_list_RC_3v=[]
    correlation_list_LR_3v=[]

    correlation_list_RC_4v_std=[]
    correlation_list_RC_3v_std=[]

    for lead_time in lead_times:

        y_validation=X_validation[1,spin+(lead_time*3):]
        y_validation=(y_validation*2)
        time_line_validation=zc.time_line_validation
        time_line_validation=time_line_validation[spin+(lead_time*3):]
        y_validation=compute_monthly_data(y_validation,time_line_validation)
        time_line_validation=np.round(time_line_validation,0)
        time_line_validation=np.unique(time_line_validation)
        time_line_validation=time_line_validation[1:-1]

        n_variables=4
        f = open(f'results_experiments/Drag{drag}RCLead{lead_time}Train{train_years}yearsValidation{validation_years}yearsAttemps{attemps}_reconstructions_{n_variables}_variables_tmp.json')
        data_RC_4v=json.load(f)
        data_RC_4v=np.array(data_RC_4v['Nino3'])

        data_LR_4v=np.load(f'results/LrDrag{drag}Lead{lead_time*3}Variables{len(variables)}.npy')

        f = open(f'results_experiments/Drag{drag}RCLead{lead_time}Train{train_years}yearsValidation{validation_years}yearsAttemps{attemps}_performances_{n_variables}_variables_tmp.json')
        data_RC_performances_4v=json.load(f)
        mean_correlation_RC_4v=np.mean(np.array(data_RC_performances_4v['correlation_TE']))
        std_correlation_RC_4v=np.std(np.array(data_RC_performances_4v['correlation_TE']))

        correlation_LR_4v=pearsonr(data_LR_4v,y_validation)

        correlation_list_RC_4v.append(mean_correlation_RC_4v)
        correlation_list_RC_4v_std.append(std_correlation_RC_4v)
        correlation_list_LR_4v.append(correlation_LR_4v.statistic)

        n_variables=3
        f = open(f'results_experiments/Drag{drag}RCLead{lead_time}Train{train_years}yearsValidation{validation_years}yearsAttemps{attemps}_reconstructions_{n_variables}_variables_tmp.json')
        data_RC_3v=json.load(f)
        data_RC_TE_3v=np.array(data_RC_3v['Nino3'])

        data_LR_3v=np.load(f'results_experiments/LrDrag{drag}Lead{lead_time*3}Variables{n_variables}.npy')

        f = open(f'results_experiments/Drag{drag}RCLead{lead_time}Train{train_years}yearsValidation{validation_years}yearsAttemps{attemps}_performances_{n_variables}_variables_tmp.json')
        data_RC_performances_3v=json.load(f)
        mean_correlation_RC_3v=np.mean(np.array(data_RC_performances_3v['correlation_TE']))
        std_correlation_RC_3v=np.std(np.array(data_RC_performances_3v['correlation_TE']))

        correlation_LR_3v=pearsonr(data_LR_3v,y_validation)

        correlation_list_RC_3v.append(mean_correlation_RC_3v)
        correlation_list_RC_3v_std.append(std_correlation_RC_3v)
        correlation_list_LR_3v.append(correlation_LR_3v.statistic)

    fig=plt.figure(figsize=(16,16))
    plt.plot(lead_times,correlation_list_LR_4v,marker="o",linewidth=10,
    label=r"LR $\tau_C$ incl.",color="orange", markeredgecolor='black',
    markeredgewidth=10,markersize=30)
    plt.plot(lead_times,correlation_list_LR_3v,marker="o",linewidth=10,label=r"LR $\tau_C$ excl.",
    color="red", markeredgecolor='black', markeredgewidth=10,
    markersize=30,linestyle="dashed")
    plt.errorbar(lead_times,correlation_list_RC_4v,correlation_list_RC_4v_std,marker="o",
    linewidth=10,label=r"RC $\tau_C$ incl.",color="blue",capsize=25, capthick=10, 
    elinewidth=10,markeredgecolor='black', markeredgewidth=10,markersize=30)
    plt.errorbar(lead_times,correlation_list_RC_3v,correlation_list_RC_3v_std,marker="o",
    linewidth=10,label=r"RC $\tau_C$ excl.",color="green",capsize=25, capthick=10, elinewidth=10,
    markeredgecolor='black', markeredgewidth=10,markersize=30,linestyle="dashed")
    plt.yticks([0.6,0.7,0.8,0.9,1.0])
    plt.xlabel("lead time [month]")
    plt.ylabel("ACC")
    plt.legend()
    plt.xticks(lead_times)
    plt.savefig(f'./imagesPaperErrorGrowth/PerformancesDrag{drag}.png',bbox_inches='tight')
    plt.show()

def plot_results_CNOPs_different_months(drag,season,delta,lead_times,
    plot_variable,plot_distribution,attemps,std):

    std_09=0.5654944835772275
    std_077=0.2920720993869393

    dictionary_months={"Jan":0,"Feb":1,
    "Mar":2,"Apr":3,"May":4,"Jun":5,"Jul":6,"Aug":7,"Sep":8,"Oct":9,"Nov":10,
    "Dec":11}

    dictionary_plot={'lead_time':[],'correlation':[],'RSE (Root Square Error)':[],'rms':[],"model":[],"TE [°C]":[], "HE [m]":[], 
                     "HW [m]":[]}
    directory_ZC='./resultsCNOPsZebiakCane_experiments'
    directory_RC='./resultsCNOPsRC_experiments'

    for month in season:

        initial_month=dictionary_months[month]

        for i,lead in enumerate(lead_times):

            file_name_ZC = f'Drag{drag}/CNOPsIm{initial_month}Delta{delta}Lead{lead}Interval[955,1000]Attemps{attemps}.json'
            file_path_ZC=os.path.join(directory_ZC,file_name_ZC)

            lead=lead*3
            file_name_RC=f'Drag{drag}/CNOPsIm{initial_month}Delta{delta}Lead{lead}Interval[950,1000]WeeklyTrueVariables3_tmp.json'

            file_path_RC=os.path.join(directory_RC,file_name_RC)

            lead=int(lead/3)
        
            with open(file_path_RC,'r') as file:
                results_RC=json.load(file)
            file.close()

            list_correlation=list(results_RC['correlation'])
            list_rms=list(results_RC['rms'])
            list_TE=list(results_RC['TE'])
            list_HW=list(results_RC["HW"])
            list_HE=list(results_RC["HE"])
            list_distance=list(results_RC['distance'])
            list_distance=[abs(list_distance[i])/std for i in range(len(list_distance))]

            if(len(list_distance)==45 and initial_month not in [2,3,4]):
                indices_to_remove=[44]
            else:
                indices_to_remove=[]
            
            for index in sorted(indices_to_remove, reverse=True):
                list_correlation.pop(index)
                list_rms.pop(index)
                list_TE.pop(index)
                list_HW.pop(index)
                list_HE.pop(index)
                list_distance.pop(index)

            
            dictionary_plot['correlation']=dictionary_plot['correlation']+list_correlation
            dictionary_plot['rms']=dictionary_plot['rms']+list_rms
            dictionary_plot['TE [°C]']=dictionary_plot['TE [°C]']+list_TE
            dictionary_plot['HE [m]']=dictionary_plot['HE [m]']+list_HE
            dictionary_plot['HW [m]']=dictionary_plot['HW [m]']+list_HW
            dictionary_plot['RSE (Root Square Error)']=dictionary_plot['RSE (Root Square Error)']+list_distance

            for i in range(len(list_distance)):
                dictionary_plot['lead_time'].append(lead)
                dictionary_plot['model'].append(r"RC $\tau_C$ excl.")
            
            lead=lead*3
            file_name_RC=f'Drag{drag}/CNOPsIm{initial_month}Delta{delta}Lead{lead}Interval[950,1000]WeeklyTrueVariables4_tmp.json'
            file_path_RC=os.path.join(directory_RC,file_name_RC)

            lead=int(lead/3)
        
            with open(file_path_RC,'r') as file:
                results_RC=json.load(file)
            file.close()

            list_correlation=list(results_RC['correlation'])
            list_rms=list(results_RC['rms'])
            list_TE=list(results_RC['TE'])
            list_HW=list(results_RC["HW"])
            list_HE=list(results_RC["HE"])
            list_distance=list(results_RC['distance'])
            list_distance=[abs(list_distance[i])/std for i in range(len(list_distance))]

            if(len(list_distance)==45 and initial_month not in [2,3,4]):
                indices_to_remove=[44]
            else:
                indices_to_remove=[]

            for index in sorted(indices_to_remove, reverse=True):
                list_correlation.pop(index)
                list_rms.pop(index)
                list_TE.pop(index)
                list_HW.pop(index)
                list_HE.pop(index)
                list_distance.pop(index)
            
            dictionary_plot['correlation']=dictionary_plot['correlation']+list_correlation
            dictionary_plot['rms']=dictionary_plot['rms']+list_rms
            dictionary_plot['TE [°C]']=dictionary_plot['TE [°C]']+list_TE
            dictionary_plot['HE [m]']=dictionary_plot['HE [m]']+list_HE
            dictionary_plot['HW [m]']=dictionary_plot['HW [m]']+list_HW
            dictionary_plot['RSE (Root Square Error)']=dictionary_plot['RSE (Root Square Error)']+list_distance

            for i in range(len(list_distance)):
                dictionary_plot['lead_time'].append(lead)
                dictionary_plot['model'].append(r"RC $\tau_C$ incl.")


            with open(file_path_ZC,'r') as file:
                results_ZC = json.load(file)
            file.close()
            
            list_distance_ZC=list(results_ZC['distance'])
            list_distance_ZC=[abs(list_distance_ZC[i])/std for i in range(len(list_distance_ZC))]

            if(len(list_distance_ZC)==45 and initial_month not in [2,3,4]):
                indices_to_remove=[44]
            else:
                indices_to_remove=[]

            if('Optimal Perturbation' in list(results_ZC.keys())):
                optimal_perturbation=list(results_ZC['Optimal Perturbation'])
                ZC_TE=[optimal_perturbation[i][0] for i in range(len(optimal_perturbation))]
                ZC_HW=[optimal_perturbation[i][1] for i in range(len(optimal_perturbation))]
                ZC_HE=[optimal_perturbation[i][2] for i in range(len(optimal_perturbation))]
            else:
                ZC_TE=list(results_ZC['TE'])
                ZC_HW=list(results_ZC['HW'])
                ZC_HE=list(results_ZC['HE'])

            for index in sorted(indices_to_remove, reverse=True):
                list_distance_ZC.pop(index)
                results_ZC['correlation'].pop(index)
                results_ZC['rms'].pop(index)
                ZC_TE.pop(index)
                ZC_HE.pop(index)
                ZC_HW.pop(index)
            
            dictionary_plot['RSE (Root Square Error)']=dictionary_plot['RSE (Root Square Error)']+list_distance_ZC
            dictionary_plot['correlation']=dictionary_plot['correlation']+results_ZC['correlation']
            dictionary_plot['rms']=dictionary_plot['rms']+results_ZC['rms']

            dictionary_plot['TE [°C]']=dictionary_plot['TE [°C]']+ZC_TE
            dictionary_plot['HE [m]']=dictionary_plot['HE [m]']+ZC_HW
            dictionary_plot['HW [m]']=dictionary_plot['HW [m]']+ZC_HE

            dictionary_plot['lead_time']+[ lead for i in range(len(list_distance_ZC))]
            dictionary_plot['model']+["ZC" for i in range(len(list_distance_ZC))]

            for i in range(len(list_distance)):
                dictionary_plot['lead_time'].append(lead)
                dictionary_plot['model'].append("ZC")


    dictionary_plot['ThS'] = [x + y for x, y in zip(dictionary_plot['HE [m]'], dictionary_plot['HW [m]'])]
    dataframe=pd.DataFrame(dictionary_plot)
    fig=plt.figure(figsize=(16,16))

    if(plot_variable=="correlation"):
        sns.boxplot(dataframe,x='lead_time',y='correlation',hue='model',linewidth=4)
    if(plot_variable=="rms"):
        sns.boxplot(dataframe,x='lead_time',y='rms',hue='model',linewidth=4)
    if(plot_variable=="rse"):

        if(drag==0.9):
            dataframe['RSE (Root Square Error)']=dataframe['RSE (Root Square Error)']/std_09
        if(drag==0.77):
            dataframe['RSE (Root Square Error)']=dataframe['RSE (Root Square Error)']/std_077
        sns.boxplot(dataframe,x='lead_time',y='RSE (Root Square Error)',hue='model',linewidth=5)
        plt.xlabel("lead time [month]")
        plt.ylabel("RSE")

        ZC_lead_3=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 3) & (dataframe['model'] == 'ZC')]
        print("median for lead time 3:{} with std:{}".format(np.percentile(ZC_lead_3,50),np.percentile(ZC_lead_3,75)-np.percentile(ZC_lead_3,25)))
        ZC_lead_6=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 6) & (dataframe['model'] == 'ZC')]
        print("median for lead time 6:{} with std:{}".format(np.percentile(ZC_lead_6,50),np.percentile(ZC_lead_6,75)-np.percentile(ZC_lead_6,25)))
        ZC_lead_9=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 9) & (dataframe['model'] == 'ZC')]
        print("median for lead time 9:{} with std:{}".format(np.percentile(ZC_lead_9,50),np.percentile(ZC_lead_9,75)-np.percentile(ZC_lead_9,25)))

        RC_lead_3_3v=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 3) & (dataframe['model'] == r"RC $\tau_C$ excl.")]
        print("mean for lead time 3:{} with std:{}".format(np.percentile(RC_lead_3_3v,50),np.percentile(RC_lead_3_3v,75)-np.percentile(RC_lead_3_3v,25)))
        RC_lead_6_3v=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 6) & (dataframe['model'] == r"RC $\tau_C$ excl.")]
        print("mean for lead time 6:{} with std:{}".format(np.percentile(RC_lead_6_3v,50),np.percentile(RC_lead_6_3v,75)-np.percentile(RC_lead_6_3v,25)))
        RC_lead_9_3v=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 9) & (dataframe['model'] == r"RC $\tau_C$ excl.")]
        print("mean for lead time 9:{} with std:{}".format(np.percentile(RC_lead_9_3v,50),np.percentile(RC_lead_9_3v,75)-np.percentile(RC_lead_9_3v,25)))
        plt.savefig(f'./imagesPaperErrorGrowth/CNOPsDrag{drag}Season{season}.png',bbox_inches='tight',dpi=300)

        RC_lead_3_4v=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 3) & (dataframe['model'] == r"RC $\tau_C$ incl.")]
        print("mean for lead time 3:{} with std:{}".format(np.percentile(RC_lead_3_4v,50),np.percentile(RC_lead_3_4v,75)-np.percentile(RC_lead_3_4v,25)))
        RC_lead_6_4v=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 6) & (dataframe['model'] == r"RC $\tau_C$ incl.")]
        print("mean for lead time 6:{} with std:{}".format(np.percentile(RC_lead_6_4v,50),np.percentile(RC_lead_6_4v,75)-np.percentile(RC_lead_6_4v,25)))
        RC_lead_9_4v=dataframe['RSE (Root Square Error)'][(dataframe['lead_time'] == 9) & (dataframe['model'] == r"RC $\tau_C$ incl.")]
        print("mean for lead time 9:{} with std:{}".format(np.percentile(RC_lead_9_4v,50),np.percentile(RC_lead_9_4v,75)-np.percentile(RC_lead_9_4v,25)))
        plt.savefig(f'./imagesPaperErrorGrowth/CNOPsDrag{drag}Season{season}.png',bbox_inches='tight',dpi=300)

    if(plot_variable=="TE"):
        sns.violinplot(dataframe,x='lead_time',y='TE [°C]',hue='model',linewidth=5,
        inner="box",width=0.8,density_norm="count",legend=False)
        plt.ylabel("NINO3 [°C]")
        plt.xlabel("lead time [month]")
        #plt.legend(loc='upper left')
        plt.savefig(f'./imagesPaperErrorGrowth/CNOPsTEDRa{drag}Season{season}.png',bbox_inches='tight',dpi=300)
    
    if(plot_variable=="HE"):
        sns.violinplot(dataframe,x='lead_time',y='HE [m]',hue='model',linewidth=4,
        inner="box",width=0.8,density_norm="count")
        plt.ylabel(r'$h_E$ [m]')
        plt.xlabel("lead time [month]")
        plt.savefig(f'./imagesPaperErrorGrowth/CNOPsHEDRa{drag}Season{season}.png',bbox_inches='tight')

    if(plot_variable=="HW"):
        sns.boxplot(dataframe,x='lead_time',y='HW [m]',hue='model',linewidth=4)

    if(plot_variable=="ThS"):
        sns.violinplot(dataframe,x='lead_time',y='ThS',hue='model',linewidth=5,
        inner="box",width=0.8,density_norm="count",legend=False)
        plt.ylabel(r'$h_E$ + $h_W$ [m]')
        plt.xlabel("lead time [month]")

        plt.savefig(f'./imagesPaperErrorGrowth/CNOPsHEDRa{drag}Season{season}.png',bbox_inches='tight')

    plt.show()

    colors = {'ZC': 'green', r"RC $\tau_C$ excl.": 'blue', r"RC $\tau_C$ incl.": 'orange'} 

    if(plot_distribution):

        selected_HE = [d for d, v in zip(dictionary_plot['HE [m]'], dictionary_plot['lead_time']) if v == 9]
        selected_HW = [d for d, v in zip(dictionary_plot['HW [m]'], dictionary_plot['lead_time']) if v == 9]
        selected_TE = [d for d, v in zip(dictionary_plot['TE [°C]'], dictionary_plot['lead_time']) if v == 9]
        
        colors_plot=[colors[label] for label,v in zip(dictionary_plot['model'],dictionary_plot['lead_time']) if v==3]

        legend_patches = [mpatches.Patch(color=colors[label], label=label) for label in colors]

        selected_TE_normalized=np.array(selected_TE)/2
        selected_HE_normalized=np.array(selected_HE)/50
        selected_HW_normalized=np.array(selected_HW)/50
        selected_full_thermocline=selected_HE_normalized+selected_HW_normalized
        fig=plt.figure(figsize=(15,15))
        plt.scatter(selected_TE_normalized,selected_full_thermocline,c=colors_plot,s=200,edgecolors='black',  
        alpha=0.8)
        plt.xlabel("NINO3")
        plt.ylabel(r"$h_E$ + $h_W$")
        legend_patches = [mpatches.Patch(color=colors[label], label=label) for label in colors]
        plt.legend(handles=legend_patches,loc='center', bbox_to_anchor=(0.5, 0.5))
        plt.savefig(f'./imagesPaperErrorGrowth/DistributionTE-HELead9Drag{drag}Season{season}.png',bbox_inches='tight')
        plt.show()

def save_reconstructions(drag,transitional_years,
train_years,validation_years,
best_parameters,lead_time,
attemps,file_name,file_name_performances,weekly,spin,variables,
normalize=True):
    
    print("##################################")
    print("\n")
    print("Analyzing lead:{}".format(lead_time))
    print("Analyzing drag:{}".format(drag))
    print("results wiil be save in the file:{}".format(file_name))
    print("performances will be saved in the file:{}".format(file_name_performances))
    print("\n")
    print("###################################")
    
    if(weekly):
        lead_time=(lead_time)*3
    
    dictionary_reconstruction={}
    input_variables_number=len(variables)+1
    zc=ZebiakCaneModel("../ZebiakCaneFiles/NoisyTrajectories_for_experiments_paper",drag,transitional_years,
    train_years,validation_years,variables,normalize)
    X_train,X_validation=zc.X_train,zc.X_validation
    time_line_validation=zc.time_line_validation
    time_line_validation_test=time_line_validation[spin+lead_time:]

    rms_min=sys.float_info.max
    performances={'correlation_TE':[]}

    for i in range(attemps):
        print("iteration:{}".format(i))

        rc=RC(**best_parameters,input_variables=input_variables_number,
        normalize=True)
        rc.train(X_train,spin,1,cycle=True) 
        reconstruction,corr_TE,_,_,rms_TE,_,_,rms_total=rc.return_full_time_series(X_validation.copy(),
        spin,time_line_validation=time_line_validation,
        cycle=True,lead=lead_time,compute_monthly=True)

        if(i==0):
            dictionary_reconstruction["Nino3"]=reconstruction[np.newaxis,:,1]
        else:
            dictionary_reconstruction["Nino3"]=np.concatenate((dictionary_reconstruction["Nino3"],
            reconstruction[np.newaxis,:,1]),axis=0)

        performances['correlation_TE'].append(corr_TE)

        print("RMS total:{}".format(rms_total))
        

        if(rms_total<rms_min):

            with open("ReservoirWeights/Drag{}BestWeightsLead{}Variables{}_tmp".format(drag,
            lead_time,len(variables)),'wb') as file:
                pickle.dump(rc,file)
            
            file.close()
            print("new best result:{}".format(rms_total))
            rms_min=rms_total
            rms_min=rms_total
            print("correlation on average TE:{}".format(corr_TE))
            print("rms TE on average:{}".format(rms_TE))
    
    dictionary_reconstruction["Nino3"]=list(dictionary_reconstruction["Nino3"].tolist())

    with open(file_name, 'w') as json_file:
        json.dump(dictionary_reconstruction,json_file)
    json_file.close()
    
    with open(file_name_performances,'w') as json_file:
        json.dump(performances,json_file)
    json_file.close()

    linear_regressor=LinearRegressor()
    linear_regressor.train(X_train,lead_time)
    predictions=linear_regressor.predict(X_validation,spin,lead_time)

    if(normalize):
        predictions=(predictions*2)
        if(weekly):
            predictions=compute_monthly_data(predictions,time_line_validation_test)
    
    np.save("results/LrDrag{}Lead{}Variables{}".format(drag,lead_time,len(variables)),predictions)

def assign_best_params(drag, n_variables):

    drag_str = str(drag).replace('.', '')
    var_name = f"best_params_{drag_str}_{n_variables}_variables"
    print("searching for best_parameters:{}".format(var_name))

    best_params = getattr(best_parameters_validation, var_name)

    return best_params




def plot_performances_SPB(drag,lead_times):

    zc=ZebiakCaneModel("../ZebiakCaneFiles/NoisyTrajectories_for_experiments_paper",drag,transitional_years=300,
    train_years=300,validation_interval=[800,1000],variables=["Nino3","ThE","ThW","Wind"],
    normalize=True)
    X_train,X_validation=zc.X_train,zc.X_validation
    time_line_validation=zc.time_line_validation
    y_validation=X_validation[1,:]

    performances_per_lead_RC_pre=[]
    performances_per_lead_RC_post=[]
    performances_per_lead_LR_pre=[]
    performances_per_lead_LR_post=[]

    for lead_time in lead_times:
        if(drag==0.9):

            with open(f"results_experiments/Drag0.9RCLead{lead_time}Train300yearsValidation[800, 1000]yearsAttemps50_reconstructions_4_variables_tmp.json", 'r') as file:
                results_RC= json.load(file)
                results_RC= np.array(results_RC["Nino3"])
                results_RC = np.percentile(results_RC,50,axis=0)
            results_LR = np.load(f"results_experiments/LrDrag0.9Lead{int(lead_time*3)}Variables4.npy")

            std=np.std(y_validation*2)
            print("std for drag:{} is:{}".format(drag,std))
            y_validation_tmp=y_validation[spin+(lead_time*3):]
            time_line_validation_tmp=time_line_validation[spin+(lead_time*3):]
            
            y_validation_tmp=y_validation_tmp*2
            y_validation_tmp=compute_monthly_data(y_validation_tmp,time_line_validation_tmp)        

            results_RC_pre=[]
            results_RC_post=[]
            results_LR_pre=[]
            results_LR_post=[]
            true_values_pre=[]
            true_values_post=[]
            for i in [2,3,4]:
                results_RC_pre.extend(results_RC[i::12])
                results_LR_pre.extend(results_LR[i::12])
                true_values_pre.extend(y_validation_tmp[i::12])
            for i in [8,9,10,11,0,1]:
                results_RC_post.extend(results_RC[i::12])
                results_LR_post.extend(results_LR[i::12])
                true_values_post.extend(y_validation_tmp[i::12])
            
            results_RC_pre=np.array(results_RC_pre)
            results_LR_pre=np.array(results_LR_pre)
            true_values_pre=np.array(true_values_pre)
            results_RC_post=np.array(results_RC_post)
            results_LR_post=np.array(results_LR_post)
            true_values_post=np.array(true_values_post)

            abs_3_RC_pre=np.mean(np.abs(results_RC_pre-true_values_pre))/std
            abs_3_RC_post=np.mean(np.abs(results_RC_post-true_values_post))/std
            abs_3_LR_pre=np.mean(np.abs(results_LR_pre-true_values_pre))/std
            abs_3_LR_post=np.mean(np.abs(results_LR_post-true_values_post))/std

            performances_per_lead_RC_pre.append(abs_3_RC_pre)
            performances_per_lead_RC_post.append(abs_3_RC_post)
            performances_per_lead_LR_pre.append(abs_3_LR_pre)
            performances_per_lead_LR_post.append(abs_3_LR_post)
        
        if(drag==0.77):

            with open(f"results_experiments/Drag0.77RCLead{lead_time}Train300yearsValidation[800, 1000]yearsAttemps50_reconstructions_3_variables_tmp.json", 'r') as file:
                results_RC= json.load(file)
                results_RC= np.array(results_RC["Nino3"])
                results_RC = np.percentile(results_RC,50,axis=0)
            results_LR = np.load(f"results_experiments/LrDrag0.77Lead{int(lead_time*3)}Variables3.npy")

            std=np.std(y_validation*2)
            print("std for drag:{} is:{}".format(drag,std))
            y_validation_tmp=y_validation[spin+(lead_time*3):]
            time_line_validation_tmp=time_line_validation[spin+(lead_time*3):]
            
            y_validation_tmp=y_validation_tmp*2
            y_validation_tmp=compute_monthly_data(y_validation_tmp,time_line_validation_tmp)        

            results_RC_pre=[]
            results_RC_post=[]
            results_LR_pre=[]
            results_LR_post=[]
            true_values_pre=[]
            true_values_post=[]
            for i in [2,3,4]:
                results_RC_pre.extend(results_RC[i::12])
                results_LR_pre.extend(results_LR[i::12])
                true_values_pre.extend(y_validation_tmp[i::12])
            for i in [8,9,10]:
                results_RC_post.extend(results_RC[i::12])
                results_LR_post.extend(results_LR[i::12])
                true_values_post.extend(y_validation_tmp[i::12])
            
            results_RC_pre=np.array(results_RC_pre)
            results_LR_pre=np.array(results_LR_pre)
            true_values_pre=np.array(true_values_pre)
            results_RC_post=np.array(results_RC_post)
            results_LR_post=np.array(results_LR_post)
            true_values_post=np.array(true_values_post)

            abs_3_RC_pre=np.mean(np.abs(results_RC_pre-true_values_pre))/std
            abs_3_RC_post=np.mean(np.abs(results_RC_post-true_values_post))/std
            abs_3_LR_pre=np.mean(np.abs(results_LR_pre-true_values_pre))/std
            abs_3_LR_post=np.mean(np.abs(results_LR_post-true_values_post))/std

            performances_per_lead_RC_pre.append(abs_3_RC_pre)
            performances_per_lead_RC_post.append(abs_3_RC_post)
            performances_per_lead_LR_pre.append(abs_3_LR_pre)
            performances_per_lead_LR_post.append(abs_3_LR_post)

    fig=plt.figure(figsize=(16,16))
    plt.plot(lead_times,performances_per_lead_LR_pre,label="LR pre SPB",
    linewidth=10,marker="o",markersize=30,markeredgecolor='black',markeredgewidth=10,
    color="orange")
    plt.plot(lead_times,performances_per_lead_LR_post,label="LR post SPB",
    linewidth=10,marker="o",markersize=30,markeredgecolor='black',markeredgewidth=10,
    color="red",linestyle="dashed")
    plt.plot(lead_times,performances_per_lead_RC_pre,label="RC pre SPB",
    linewidth=10,marker="o",markersize=30,markeredgecolor='black',
    markeredgewidth=10,color="blue") 
    plt.plot(lead_times,performances_per_lead_RC_post,label="RC post SPB",
    linewidth=10,marker="o",markersize=30,markeredgecolor='black',
    markeredgewidth=10,color="green",linestyle="dashed")
    plt.xticks(lead_times)
    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    plt.legend()
    plt.ylabel("MAE")
    plt.xlabel("lead time [month]")
    plt.savefig(f'./imagesPaperErrorGrowth/PerformancesSPBDrag{drag}RCLRComparison.png',bbox_inches='tight')

    plt.show()

    print("performances pre LR 3 months lead {}".format(performances_per_lead_LR_pre[0]))
    print("performances pre LR 6 months lead {}".format(performances_per_lead_LR_pre[1]))
    print("performances pre RC 3 months lead {}".format(performances_per_lead_RC_pre[0]))
    print("difference pre RC 6 months lead {}".format(performances_per_lead_RC_pre[1]))
    

    return 

spin=180

#plot the ACC RC performances based on the results of our experiments contained in the directory results_experiments

#plot_performances(drag=0.77,
#transitional_years=transitional_years,
#train_years=train_years,
#validation_years=validation_years,spin=spin,
#lead_times=[3,6,9,12,15,18])

#plot the seasonal performances taing the results from the directory results_experiments containing the results of our experiments in terms of performances
#plot_performances_SPB(drag=0.77,lead_times=[3,6,9,12,15,18])

#plot the CNOPs distribution for the distance and different variables , taking the results from the directory results_CNOPsRC_experiments
#and the directory resultsCNOPsZebiakCane_experiments containing the results of out experiments
#plot_results_CNOPs_different_months(drag=0.9,
#season=["Dec","Jan","Feb"],delta=0.05,
#lead_times=[3,6,9],plot_variable="rse",
#attemps=10,plot_distribution=True,std=1)

#save reconstructions and performances and save results in results and best weights reservoir in ReservoirWeights
#for drag in [0.77,0.9]:
#    drag=drag

#    for variables in [["Nino3","ThE","ThW","Wind"],["Nino3","ThE","ThW"]]:

#        n_variables=len(variables)

        # Call the function with the parsed arguments
#        best_params = assign_best_params(drag=drag, n_variables=n_variables)

#        for lead_time in [3,6,9,12,15,18]:
#            file_name_reconstructions="results/Drag{}RCLead{}Train{}yearsValidation{}yearsAttemps{}_reconstructions_{}_variables_tmp.json".format(
#            drag,lead_time,train_years,validation_years,attemps,n_variables)
#            file_name_performances="results/Drag{}RCLead{}Train{}yearsValidation{}yearsAttemps{}_performances_{}_variables_tmp.json".format(
#            drag,lead_time,train_years,validation_years,attemps,n_variables)
#            save_reconstructions(drag,transitional_years,train_years,
#            validation_years,best_params,
#            lead_time,attemps,file_name_reconstructions,file_name_performances,
#            variables=variables,weekly=True,spin=spin)

