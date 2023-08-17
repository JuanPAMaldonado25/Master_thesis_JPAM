import datetime #to synchronize data
#treating the timestamps as time objects instead of floats
start_time = datetime.datetime(2023,3,24,00,00,00)  #Defining t=0 as March 24th 10:00:00 2023
noise_boundaries = [80,280] #y limits for the noise tempeature in plot 1
Tsensor_boundaries = [15,25] #y limits for the sensor temperature in plot 1
common_path='C:/Madmax/MORPURGO/' #The directory that contains the folders where the new files are stored
longtime_DAQ_path = 'Long_Term_SA/'
measurement_tracked = "CB100_2023*" #the folders that store the files have the same name
calibrations_path = 'Calib_SA/40MHz/' #where all the calibration folders can be found
calibration1 = 'calibration_20230323/'
calibration2 = 'calibration_20230406/'
filter_window_raw = 2001
filter_window_avg = 1001 #must be odd
bin_bandwidth = 1e3 #Hz
time_per_file = 0.75e-3*6*400 #seconds per 6 loops of 400 averages i.e. 0.75 ms per sweep
exclude_points_raw = 1000
exclude_points_avg = 500
change_of_time = datetime.datetime(2023,3,26,2,00,00)

#we have two calibrations, here, I define when to change from cal 1 to cal 2
first_cal_valid_until = datetime.datetime(2023,4,5,14,30,40) #timestamp of final file. After this one, the files are calibrated with the second set

#Changing and synchronizing x-axis with timestamps
#reading all the files inside temperature, B field and DAQ files
import threading, msvcrt #to create the timed input for the shifter
import sys #to safely exit the code
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') #ideal for non interacive plots to be saved in png
plt.ioff() #turning off interacting mode consumes less memory
import numpy as np
import seaborn as sns #to style the plot
from pytimedinput import timedInput
from email.message import EmailMessage #to create email message
import ssl  #to add security to the email
import smtplib #to send email
import gc #garbage collector

sns.set_style("whitegrid")

GOLDENRATIO = (1 + 5 ** 0.5) / 2
GOLDEN = (16, 16/GOLDENRATIO)

import pandas as pd #to manage datafiles

import glob,os #to read files

params = {'legend.fontsize': 12,
         'axes.labelsize': 15,
         'axes.titlesize':20,
         'xtick.labelsize':12,
         'ytick.labelsize':12}

plt.rcParams.update(params)

from scipy.signal import savgol_filter #to create baseline
from scipy.optimize import curve_fit #to fit gaussian
from sys import exit #to exit code
import time #to pause between iterations


#Reading necessary files to Y factor calibrate the incoming data---------------
#This block is run only once with the calibration data taken at 40MHz bandwidth with the S.A.

#For the baseline and diodes to obtain the Y factor we use the final average files

#-------------------------------FIRST CALIBRATION----------------------------------------------------------------------------------------------------------------------------------------------------------
baseline_meas_cal1 =   pd.read_csv(common_path+calibrations_path+calibration1+'spareLNA_Baseline_20230323_191946/rawfft/spareLNA_Baseline_20230323_193110_S01a'+'.smp',sep='\t',header=0,names=['Frequency (Hz)','baseline (W)'])
baseline_meas_cal1.set_index('Frequency (Hz)',inplace=True)

diode_OFF_meas_cal1 =   pd.read_csv(common_path+calibrations_path+calibration1+'spareLNA_NoiseDiodeOff_20230323_183443/rawfft/spareLNA_NoiseDiodeOff_20230323_184606_S01a'+'.smp',sep='\t',header=0,names=['Frequency (Hz)','diode_OFF (W)'])
diode_OFF_meas_cal1.set_index('Frequency (Hz)',inplace=True)


diode_ON_meas_cal1 =   pd.read_csv(common_path+calibrations_path+calibration1+'spareLNA_NoiseDiodeOn_20230323_185054/rawfft/spareLNA_NoiseDiodeOn_20230323_190219_S01a'+'.smp',sep='\t',header=0,names=['Frequency (Hz)','diode_ON (W)'])
diode_ON_meas_cal1.set_index('Frequency (Hz)',inplace=True)

standards_40MHz_cal1 = pd.concat([baseline_meas_cal1,diode_OFF_meas_cal1,diode_ON_meas_cal1],axis=1)

#subtracting the baseline
standards_40MHz_cal1['diode_OFF (W) no_baseline'] = standards_40MHz_cal1['diode_OFF (W)'] - standards_40MHz_cal1['baseline (W)']
standards_40MHz_cal1['diode_ON (W) no_baseline'] = standards_40MHz_cal1['diode_ON (W)'] - standards_40MHz_cal1['baseline (W)']
standards_40MHz_cal1['Y factor'] = standards_40MHz_cal1['diode_ON (W) no_baseline']/standards_40MHz_cal1['diode_OFF (W) no_baseline']

#------------------------SECOND CALIBRATION----------------------------------------------------------------------------------------------------------------------------------------------------------------

baseline_meas_cal2 =   pd.read_csv(common_path+calibrations_path+calibration2+'20230406_40mhz_baseline01b50'+'.csv',sep=',',header=44,names=['Frequency (Hz)','baseline (W)'])
baseline_meas_cal2.set_index('Frequency (Hz)',inplace=True)

diode_OFF_meas_cal2 =   pd.read_csv(common_path+calibrations_path+calibration2+'20230406_40MHz_diodeOFF'+'.csv',sep=',',header=44,names=['Frequency (Hz)','diode_OFF (W)'])
diode_OFF_meas_cal2.set_index('Frequency (Hz)',inplace=True)


diode_ON_meas_cal2 =   pd.read_csv(common_path+calibrations_path+calibration2+'20230406_40MHz_diodeON'+'.csv',sep=',',header=44,names=['Frequency (Hz)','diode_ON (W)'])
diode_ON_meas_cal2.set_index('Frequency (Hz)',inplace=True)

standards_40MHz_cal2 = pd.concat([baseline_meas_cal2,diode_OFF_meas_cal2,diode_ON_meas_cal2],axis=1)

#subtracting the baseline
standards_40MHz_cal2['diode_OFF (W) no_baseline'] = standards_40MHz_cal2['diode_OFF (W)'] - standards_40MHz_cal2['baseline (W)']
standards_40MHz_cal2['diode_ON (W) no_baseline'] = standards_40MHz_cal2['diode_ON (W)'] - standards_40MHz_cal2['baseline (W)']
standards_40MHz_cal2['Y factor'] = standards_40MHz_cal2['diode_ON (W) no_baseline']/standards_40MHz_cal2['diode_OFF (W) no_baseline']

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_Bfield(folder='C:/Madmax/MORPURGO/Gauss/Lakeshore/*.dat'):
    B_files = sorted(glob.glob(folder))
    B_total = []
    for B_file in sorted(B_files):
        _,B_temp=np.loadtxt(B_file,delimiter='\t', unpack=True)
        B_total.extend(B_temp)
    return B_total

def get_time_days(parameter,folder,start_time = start_time):

    ntime_shifted_complete = []
    if parameter == 'Noise T':
        conversion = [ (datetime.datetime(int(str(i)[0:4]),int(str(i)[4:6]),int(str(i)[6:8]),int(str(i)[8:10]),int(str(i)[10:12]),int(str(i)[12:14])) - start_time).total_seconds()/(3600*24) for i in folder]
        return conversion
    if parameter=='B field':
        B_total = []
        files = sorted(glob.glob(folder))
        for file in sorted(files):
            ntime_temp,B_temp=np.loadtxt(file,delimiter='\t', unpack=True)
            ntime_shifted_complete.extend(ntime_temp)
            B_total.extend(B_temp)
        conversion =  [(datetime.datetime(int(str(i)[0:4]),int(str(i)[4:6]),int(str(i)[6:8]),int(str(i)[8:10]),int(str(i)[10:12]),int(str(i)[12:14])) - start_time).total_seconds()/(3600*24) for i in ntime_shifted_complete]
        return B_total,conversion
            
    if parameter=='Temperature':
        files = sorted(glob.glob(folder))
        YYYYMMDDHHMMSS_list = []
        T_LNA_list = []
        T_booster_list = []
        for file in files:
            T_sensors = pd.read_csv(file,sep='\t',names=['dunno','date','hour','T1_SA','T2_Rail','T3_booster','T4_LNA','T5_control'])
            T_sensors.drop(['dunno'],axis=1,inplace=True)
            T_LNA_list.extend(list(T_sensors['T4_LNA']))
            T_booster_list.extend(list(T_sensors['T3_booster']))
            T_sensors['datetime'] = T_sensors['date']+' '+T_sensors['hour']
            datetime_object = pd.to_datetime(T_sensors['datetime'])
            YYYYMMDDHHMMSS_list.extend(datetime_object)
        conversion = [(YYYYMMDDHHMMSS_list[single_datetime]  - datetime.timedelta(hours=1) - start_time).total_seconds()/(3600*24) if (YYYYMMDDHHMMSS_list[single_datetime]<change_of_time) else (YYYYMMDDHHMMSS_list[single_datetime]  - datetime.timedelta(hours=2) - start_time).total_seconds()/(3600*24) for single_datetime in range(len(YYYYMMDDHHMMSS_list))]
        return T_LNA_list,T_booster_list,files,conversion

def send_alert(body='Ignore this message'):
    email_sender = 'madmaxmonitoring2023@gmail.com'
    email_pwd = 'wkdmgnnaududzokj'#'omoee2023'
    #email_receiver = 'jparcilam@unal.edu.co'
    shifters=['dkreike@mpp.mpg.de','jparcilam@unal.edu.co','maldonad@mpp.mpg.de','bernardo.ary@rwth-aachen.de','vijay.dabhi@etu.univ-amu.fr','bela@mpp.mpg.de','pralavor@cppm.in2p3.fr','dstrom@mpp.mpg.de']
    subject = 'MADMON Alert'


    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = ', '.join(shifters)
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context() #to add security to the email

    #sending email
    with smtplib.SMTP_SSL('smtp.gmail.com',465,context = context) as smtp:
        #logging in and sending:
        smtp.login(email_sender,email_pwd)
        smtp.sendmail(email_sender,shifters,em.as_string())

    return None

def Power_to_noiseT(raw_file_address,timestamp,z_amp=40,T2=297):
    '''
    Converts the y axis from power scale to Noise temperature scale with baseline removed by using the Y factor calibration

    input: list of 4 filenames corresponding to the 4 files 1 per sample. Each one contains the y values of the dataset in Watts directly measured from the SA. Optional parameters are the cold load
    temperature and the impedance of the amplifier

    return: the dataset in Kelvin units for noise temperature, obtained by performing a Y factor calibration
    '''

    #Reading the datafile into a pandas dataframe

    dataframe_in_Watts =   pd.read_csv(raw_file_address[0],sep='\t',header=0,names=['Frequency (Hz)','Power (W)'])
    dataframe_in_Watts.set_index('Frequency (Hz)',inplace=True)
    for raw_path in raw_file_address[1:]:
        temp_dataframe =   pd.read_csv(raw_path,sep='\t',header=0,names=['Frequency (Hz)','Power (W)'])
        temp_dataframe.set_index('Frequency (Hz)',inplace=True)
        dataframe_in_Watts = pd.concat((dataframe_in_Watts,temp_dataframe),axis='columns')

    dataframe_in_Watts['mean_samplers'] = dataframe_in_Watts.mean(axis=1)
    #Y factor calibration-----------
    T2 = 297 #Cold load is the load at room temperature
    z_amp = 40 #mismatch factor comes from the impedance of the LNA
    gamma = (50-z_amp)/(50+z_amp)
    enr_19 = 22.37-30 #ENR reported by the noise diode - 30dB from the attenuator
    noise_diode_19 = T2*0.172584 #rf.db10_2_mag(enr_19) #value reported at 19GHz
    T1_19 = T2+noise_diode_19 #the hot load is the room temperature plus the noise coming by the diode

    #Obtaining timestamp to decide which calibration to use
    if True:#timestamp <= first_cal_valid_until:
        diode_off=standards_40MHz_cal1['diode_OFF (W) no_baseline']
        y_factor_mean = np.mean(standards_40MHz_cal1['Y factor'])
    elif False:#timestamp > first_cal_valid_until: #by putting elif instead of else I ensure that no false positives are sent to the calibration 2
        print(timestamp)
        diode_off=standards_40MHz_cal2['diode_OFF (W) no_baseline']
        y_factor_mean = np.mean(standards_40MHz_cal2['Y factor'])
    

    T_e = ((T1_19 - y_factor_mean*T2)/(y_factor_mean-1))*(1-gamma)/(1+gamma) #temperature of the amplifier
    T_load_amp = T_e*(1+gamma) + T2*(1-gamma) #temperature of the load with amplifier, this is our calibration parameter for the unit conversion
    #-------------

    dataframe_in_Kelvin = pd.DataFrame(dataframe_in_Watts['mean_samplers']*T_load_amp/diode_off.values.mean())
    dataframe_in_Kelvin.rename(columns={'mean_samplers':'Temperature (K)'},inplace=True)
    return dataframe_in_Kelvin

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def avg_noise_bins(data,number_of_bins=5):

    '''
    This function divides the frequency span in several bins and computes the average noise temperature per bin. The number of bins b 
    should be such that len(dataset)%b = 0

    input: datafile to be used. For CERN 2023, the booster 40MHz measurement after conversion to noise temperature units
            should be a pandas dataframe
    output: arrays containing the temperature per bin  and avg frequency per bin in a 2 column dataframe
    '''

    binned_data = np.asarray(data).reshape((int(len(data)/number_of_bins),number_of_bins),order='F')
    binned_frequency = np.asarray(data.index).reshape((int(len(data.index)/number_of_bins),number_of_bins),order='F')

    avg_temperatures = np.mean(binned_data,axis=0)
    avg_frequency = np.mean(binned_frequency,axis=0)
    temp =pd.DataFrame(np.stack((avg_frequency,avg_temperatures),axis=1),columns=['Frequency (Hz)','Temperature (K)'])
    temp.set_index('Frequency (Hz)',inplace=True)
    return temp

def expectation(t,initial_value):
    return [(1/(bin_bandwidth*60))*np.power(initial_value,-2)*i for i in t]

def save_data():
    try:
        #saving booster peak evolution for raw datafiles
        boosterPeak_T_evolution_dataframe_raw = pd.DataFrame(boosterPeak_T_evolution_raw,columns=['Time stamp','Temperature (K)'])
        boosterPeak_T_evolution_dataframe_raw.set_index('Time stamp',inplace=True)
        boosterPeak_T_evolution_dataframe_raw.to_csv('monitoring_data/boosterPeak_T_evolution_raw.csv', sep='\t')
        
        #saving booster peak evolution for averaged datafiles
        boosterPeak_T_evolution_dataframe_avg = pd.DataFrame(boosterPeak_T_evolution_avg,columns=['Time stamp','Temperature (K)'])
        boosterPeak_T_evolution_dataframe_avg.set_index('Time stamp',inplace=True)
        boosterPeak_T_evolution_dataframe_avg.to_csv('monitoring_data/boosterPeak_T_evolution_avg.csv', sep='\t')

        #saving mean and std for raw datafiles
        mean_sigma_dataframe_raw = pd.DataFrame(mean_sigma_list_raw,columns=['Time stamp','Mean','Rms','fit Sigma'])
        mean_sigma_dataframe_raw.set_index('Time stamp',inplace=True)
        mean_sigma_dataframe_raw.to_csv('monitoring_data/mean_sigma_raw.csv', sep='\t')

        #saving mean and std for avg datafiles
        mean_sigma_dataframe_avg = pd.DataFrame(mean_sigma_list_avg,columns=['Time stamp','Mean','Rms','fit Sigma'])
        mean_sigma_dataframe_avg.set_index('Time stamp',inplace=True)
        mean_sigma_dataframe_avg.to_csv('monitoring_data/mean_sigma_avg.csv', sep='\t')

        #saving avg temperature per bin raw datafiles
        TperBin_dataframe_raw = pd.DataFrame(TperBin_list_raw,columns=['Time stamp','Bin 1','Bin 2','Bin 3','Bin 4','Bin 5'])
        TperBin_dataframe_raw.set_index('Time stamp',inplace=True)
        TperBin_dataframe_raw.to_csv('monitoring_data/TperBin_raw.csv', sep='\t')

        #saving avg temperature per bin avg datafiles
        TperBin_dataframe_avg = pd.DataFrame(TperBin_list_avg,columns=['Time stamp','Bin 1','Bin 2','Bin 3','Bin 4','Bin 5'])
        TperBin_dataframe_avg.set_index('Time stamp',inplace=True)
        TperBin_dataframe_avg.to_csv('monitoring_data/TperBin_avg.csv', sep='\t')

        success=True
    except:
        success=False

    return success

def add_subplot_axes(ax,rect,facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    #subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def MADMON_plot():
    plt.figure(figsize=GOLDEN)
    ax1 = plt.subplot(3,1,1) #there are three rows, this row has 1 plot, and the first plot
    plt.title('Temperature evolution, Total (corrupt) files = '+str(file_counter)+' ('+'{:0.1%}'.format(zero_files_counter/file_counter)+') / Lifetime w B (min) = '+
              str(np.round((1-(zero_files_counter/file_counter)-(number_0field/(len(B_gaussmeter)-rampup)))*file_counter*time_per_file/60,2)),fontsize=15)
    _,boosterPeak_T_evolution_y_raw = zip(*boosterPeak_T_evolution_raw)
    _,boosterPeak_T_evolution_y_avg = zip(*boosterPeak_T_evolution_avg)
    ax1.plot(time_noiseT,boosterPeak_T_evolution_y_raw,color='black',label='Raw')
    ax1.plot(time_noiseT,boosterPeak_T_evolution_y_avg,color='red',label='Average')
    ax1.set_xlim([0,25])
    ax2  = ax1.twinx()
    ax2.plot(time_Tsensors,T_LNA_list,color='green',label='LNA',marker='x',alpha=0.5,markersize=2)
    ax2.plot(time_Tsensors,T_booster_list,color='blue',label='Booster',marker='x',alpha=0.5,markersize=2)
    ax1.set_ylabel('Noise Temperature (K)')
    ax2.set_ylabel('Sensor Temperature ( C)')
    ax1.set_xlabel('Days since '+str(start_time))
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax1.set_ylim(noise_boundaries)
    ax2.set_ylim(Tsensor_boundaries)
    ax2.grid(b=False,axis='both')

    ax4 = plt.subplot(3,3,4)
    plt.title('T per 1 kHz, '+'{12}{13}/{10}{11}/{6}{7}{8}{9},{15}{16}:{17}{18}'.format(*str(data_label_raw))+' UTC',fontsize=15)
    plt.plot(raw_booster_meas_T.index/1e9,raw_booster_meas_T['Temperature (K)'],alpha=0.3,label=data_label_raw)
    plt.plot(raw_booster_meas_T_perBin.index/1e9, raw_booster_meas_T_perBin['Temperature (K)'],color='red',marker='x',linestyle='None',label='8MHz T avg')
    plt.plot(raw_booster_meas_T.index[exclude_points_raw:-exclude_points_raw]/1e9,baseline_savgol_raw,color='black',label='filter')
    plt.ylabel('raw noise T(K)')
    plt.xlabel('Frequency (GHz)')
    plt.xticks([raw_booster_meas_T.index[-1]/1e9]+list(raw_booster_meas_T.index[::int(np.floor(len(raw_booster_meas_T.index)/5))]/1e9))
    plt.grid(b=True,color='red',linestyle='--',axis='x')
    ax_noise_raw = add_subplot_axes(ax4,[0.35,0.55,0.3,0.3])
    ax_noise_raw.plot(avg_booster_meas_T.index[exclude_points_raw:-exclude_points_raw]/1e9,booster_no_baseline_raw/popt_raw[2],alpha=0.6)
    ax_noise_raw.grid(b=False,axis='x')
    ax_noise_raw.grid(b=True,axis='y')
    ax_noise_raw.set_xticks([])


    ax5 = plt.subplot(3,3,5)
    plt.title('T per 1 kHz')
    plt.plot(avg_booster_meas_T.index/1e9,avg_booster_meas_T['Temperature (K)'],alpha=0.3,label=data_label_raw)
    plt.plot(avg_booster_meas_T_perBin.index/1e9, avg_booster_meas_T_perBin['Temperature (K)'],color='red',marker='x',linestyle='None',label='8MHz T avg')
    plt.plot(avg_booster_meas_T.index[exclude_points_avg:-exclude_points_avg]/1e9,baseline_savgol_avg,color='black',label='filter',linewidth=0.8)
    plt.ylabel('avg noise T(K)')
    plt.xlabel('Frequency (GHz)')
    plt.xticks([avg_booster_meas_T.index[-1]/1e9]+list(avg_booster_meas_T.index[::int(np.floor(len(avg_booster_meas_T.index)/5))]/1e9))
    plt.grid(b=True,color='red',linestyle='--',axis='x')
    ax_noise_avg = add_subplot_axes(ax5,[0.35,0.55,0.3,0.3])
    ax_noise_avg.plot(avg_booster_meas_T.index[exclude_points_avg:-exclude_points_avg]/1e9,booster_no_baseline_avg/popt_avg[2],alpha=0.6)
    ax_noise_avg.grid(b=False,axis='x')
    ax_noise_avg.grid(b=True,axis='y')
    ax_noise_avg.set_xticks([])

    ax6 = plt.subplot(3,3,6)
    plt.title('Gaussmeter, w/o B field = '+'{:0.1%}'.format(number_0field/(len(B_gaussmeter)-rampup)))
    ax6.plot(time_gaussmeter,B_gaussmeter,label='Booster',color='black',)
    ax6.set_ylabel('B Field (T)')
    ax6.yaxis.set_label_position("right")
    ax6.yaxis.tick_right()
    ax6.set_xlabel('Days since ' + str(start_time))
    ax6.set_xlim([0,25])
    #ax_zoom = add_subplot_axes(ax6,[0.5,0.35,0.4,0.4])
    #ax_zoom.plot(time_gaussmeter[-6*360:],B_gaussmeter[-6*360:],color='black',label='Last 6 hours')
    #ax_zoom.grid(b=False)
    #ax_zoom.legend(fontsize=8)

    ax7 = plt.subplot(3,3,7)
    plt.title('Mean= '+str(np.round(popt_raw[1],4))+' , $\sigma$= '+str(np.abs(np.round(popt_raw[2],4))))
    plt.plot(bin_centers_raw/popt_raw[2],hist_values_raw,'b+:',label='last raw')
    plt.plot(bin_centers_raw/popt_raw[2],gaus(bin_centers_raw,*popt_raw),'r',label='Gaussian')
    ax7.text(0, 1, r'$R^2$ = '+str(np.round(r2_raw_list[-1],3)), fontsize=12)
    ax7.text(0, 10, r'RMS = '+str(np.round(sigma_raw,4)), fontsize=12)
    plt.legend(loc='upper left')
    plt.xlabel(r'Residuals [$\sigma$]')
    plt.ylabel('Number of bins')
    plt.yscale('log')
    plt.gca().set_ylim(bottom=0.9)
    #plt.xlim([-0.02,0.02]) #by inspection, might have to be changed, but it is fixed just to ensure proper visual comparison in the plot evolution

    ax8 = plt.subplot(3,3,8)
    plt.title('Mean= '+str(np.round(popt_avg[1],4))+', $\sigma$= '+str(np.abs(np.round(popt_avg[2],5))))
    plt.plot(bin_centers_avg/popt_avg[2],hist_values_avg,'b+:',label='avg')
    plt.plot(bin_centers_avg/popt_avg[2],gaus(bin_centers_avg,*popt_avg),'r',label='Gaussian')
    ax8.text(0, 1, r'$R^2$ = '+str(np.round(r2_avg_list[-1],3)), fontsize=12)
    ax8.text(0, 10, r'RMS = '+str(np.round(sigma_avg,5)), fontsize=12)
    plt.legend(loc='upper left')
    plt.xlabel(r'Residuals [$\sigma$]')
    plt.ylabel('Number of bins')
    plt.yscale('log')
    plt.gca().set_ylim(bottom=0.9)
    #plt.xlim([-0.02,0.02])

    ax9 = plt.subplot(3,3,9)
    plt.title(r'$\sigma^{-2}$'+' evolution')
    ax9.plot(time_noiseT,1/(bin_bandwidth*np.asarray(sigma_list_raw)**2),label='raw',color='black')
    ax9.legend(loc='upper left')
    ax9.set_ylabel('Lifetime per file (s)')
    ax9.set_xlabel('Days since '+str(start_time))
    ax9.grid(b=False,axis='y')
    ax9.set_xlim([0,25])

    ax10 = ax9.twinx()
    sigma_raw_initial = sigma_list_raw[0]
    ax10.plot(time_noiseT,(1-number_0field/(len(B_gaussmeter)-rampup))*np.asarray(expectation(range(len(sigma_list_avg)),sigma_raw_initial)),color='blue',label='expectation',alpha=0.7)
    ax10.plot(time_noiseT,(1-number_0field/(len(B_gaussmeter)-rampup))/(60*bin_bandwidth*(np.asarray(sigma_list_avg)**2)),label='avg',color='red')
    ax10.legend(loc='lower right')
    ax10.tick_params(axis='y', colors='red') #tick colors
    ax10.spines['right'].set_color('red') #spine color
    ax10.set_ylabel('Lifetime w B (min)')
    ax10.yaxis.set_label_position("right")
    ax10.yaxis.tick_right()

    plt.subplots_adjust(hspace=.6)
   
    plt.savefig('monitoring_images/control_image.png',dpi=1200,format='png')
    
    plt.clf() #clears figure to save memory

    plt.close('all')
    gc.collect()
    return None
def running_avg(term_n,N,last_avg):
    """
    Computes the running average from start to file N. This replaces the current DAQ avg computation and no longer relies on a single run to obtain the total average from t=0 to t=now
    term_n = last raw datafile read, after converting to noise temperature
    N = counter: number of files, including the last raw file written to disk.
    last_avg = averaged computed in the previous iteration. For the first iteration, last_avg = 0, and because N=0 then the new last_avg is equal to term_n, as expected
    """
    return term_n/N + ((N-1)/N)*last_avg

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#the code below only executes when running the python file as a script, but not when importing it as a module
if __name__ == "__main__":

    print('-------MADMAX data MONitoring initialized :D ------------')
    boosterPeak_T_evolution_raw = [] #Initializing lists to save data
    boosterPeak_T_evolution_avg = [] #Initializing lists to save data
    mean_sigma_list_raw = []
    mean_sigma_list_avg = []
    TperBin_list_raw = []
    TperBin_list_avg = []
    sigma_list_raw = [] #redundant but useful for plot
    sigma_list_avg = [] #redundant but useful for plot
    T_LNA_list = []
    T_booster_list = []
    r2_raw_list = []
    r2_avg_list = []
    timestamp_list = []
    boolean_saving = int(True) #to save only the last 2 plots. 
    sample_counter = 0 #initializing counter
    new_data_from_DAQ_counter = 0
    zero_files_counter = 0
    test_counter = 0
    
    #First, the computation is done for all the previously existing files---------------------------------------------------------------------------------------------

    #read all files written
    new_datasets_raw_total = sorted(glob.glob(common_path+longtime_DAQ_path+measurement_tracked+"/rawfft/*.smp")) 

    #convert the y axis to noise temperature and obtain the mean noise T per bin
    file_counter=0 #includes both corrupt and noncorrupt
    noncorrupt_file_counter = 0
    last_avg = 0 #initializing the running average

    #timing the time taken to analyze the existing data
    start = time.time()
    for i in range(int(len(new_datasets_raw_total)/4)):
        file_counter+=1 
        if i%50 == 0:
            print('Monitored '+str(i)+' files out of '+str(int(len(new_datasets_raw_total)/4)))
        sample_counter+=1
        new_datasets_raw = new_datasets_raw_total[i*4:(i+1)*4]
        timestamp = int(os.path.basename(new_datasets_raw[0])[6:-9])
        raw_booster_meas_T = Power_to_noiseT(new_datasets_raw,timestamp) #have to do it before the if, because the condition is based on the output of this function
        
        if any(raw_booster_meas_T.index != 0)and(any(raw_booster_meas_T.values) != 0): #only proceed if this is not a 0 file
            noncorrupt_file_counter+=1
            timestamp_list.append(timestamp)
        
            time_noiseT = get_time_days(parameter='Noise T',folder = timestamp_list)
            raw_booster_meas_T_perBin = avg_noise_bins(raw_booster_meas_T)

            avg_booster_meas_T = running_avg(raw_booster_meas_T,noncorrupt_file_counter,last_avg)
            avg_booster_meas_T_perBin = avg_noise_bins(avg_booster_meas_T)

            middle_point_raw =int(np.round(len(raw_booster_meas_T.index)/2,0))
            mean_boosterPeak_raw=raw_booster_meas_T.iloc[middle_point_raw-10:middle_point_raw+10].mean()


            middle_point_avg =int(np.round(len(avg_booster_meas_T.index)/2,0))
            mean_boosterPeak_avg=avg_booster_meas_T.iloc[middle_point_avg-10:middle_point_avg+10].mean()


            #we create the baseline by smoothing with a savgolay filter
            try:
                baseline_savgol_raw = savgol_filter(raw_booster_meas_T['Temperature (K)'],filter_window_raw,2,mode='constant')[exclude_points_raw:-exclude_points_raw]
                baseline_savgol_avg = savgol_filter(avg_booster_meas_T['Temperature (K)'],filter_window_avg,2,mode='constant')[exclude_points_avg:-exclude_points_avg]
            except:
                print('Could not compute baseline')


            #we remove the baseline by dividing the original dataset to the sav golay smooth
            booster_no_baseline_raw = pd.DataFrame((-1 + raw_booster_meas_T['Temperature (K)'].iloc[exclude_points_raw:-exclude_points_raw]/baseline_savgol_raw))
            booster_no_baseline_avg = pd.DataFrame((-1 + avg_booster_meas_T['Temperature (K)'].iloc[exclude_points_avg:-exclude_points_avg]/baseline_savgol_avg))

            #creating a histogram of the y values to check gaussianity
            try:
                hist_values_raw, hist_binEdges_raw = np.histogram(booster_no_baseline_raw['Temperature (K)'],bins='auto')
                hist_values_avg, hist_binEdges_avg = np.histogram(booster_no_baseline_avg['Temperature (K)'],bins='auto')
            except:
                print("Could not compute histograms on file "+str(timestamp))

            #name of the files to use as labels and to save the figures, both variables should actually match
            data_label_raw = os.path.basename(os.path.commonprefix(new_datasets_raw))[:-3] #path basename takes only the filename, not the address. common prefix takes the common name, [:-3] removes _S0
            

            #computing mean and std as initial guess for the gaussian fit
            mean_raw = booster_no_baseline_raw['Temperature (K)'].mean() #initial guess for the fit
            mean_avg = booster_no_baseline_avg['Temperature (K)'].mean()
                            
            sigma_raw = booster_no_baseline_raw['Temperature (K)'].std() #initial guess for the fit

            sigma_avg = booster_no_baseline_avg['Temperature (K)'].std()

            #bin centers to plot gaussian fit
            bin_centers_raw = 0.5*(hist_binEdges_raw[1:] + hist_binEdges_raw[:-1])
            bin_centers_avg = 0.5*(hist_binEdges_avg[1:] + hist_binEdges_avg[:-1])


            try:
                #doing a gaussian fit
                popt_raw,pcov_raw = curve_fit(gaus,bin_centers_raw,hist_values_raw,p0=[1,mean_raw,sigma_raw])
                popt_avg,pcov_avg = curve_fit(gaus,bin_centers_avg,hist_values_avg,p0=[1,mean_avg,sigma_avg])
                fit_sigma_raw = popt_raw[-1]
                fit_sigma_avg = popt_avg[-1]

                residuals_raw = hist_values_raw - gaus(bin_centers_raw, *popt_raw)
                ss_res_raw = np.sum(residuals_raw**2)
                ss_tot_raw = np.sum((hist_values_raw-np.mean(hist_values_raw))**2)
                r2_raw = 1 - (ss_res_raw / ss_tot_raw)

                residuals_avg = hist_values_avg - gaus(bin_centers_avg, *popt_avg)
                ss_res_avg = np.sum(residuals_avg**2)
                ss_tot_avg = np.sum((hist_values_avg-np.mean(hist_values_avg))**2)
                r2_avg = 1 - (ss_res_avg / ss_tot_avg)

                r2_raw_list.append(r2_raw)
                r2_avg_list.append(r2_avg)

            except:
                pass

            TperBin_list_raw.append(np.insert(raw_booster_meas_T_perBin['Temperature (K)'].values,0,timestamp,axis=0))
            TperBin_list_avg.append(np.insert(avg_booster_meas_T_perBin['Temperature (K)'].values,0,timestamp,axis=0)) #where to insert, index to insert, value to insert, axis
            boosterPeak_T_evolution_raw.append([timestamp,float(mean_boosterPeak_raw.values)])
            boosterPeak_T_evolution_avg.append([timestamp,float(mean_boosterPeak_avg.values)])
            sigma_list_raw.append(sigma_raw)
            mean_sigma_list_raw.append([timestamp,mean_raw,sigma_raw,np.abs(fit_sigma_raw)])
            sigma_list_avg.append(sigma_avg)
            mean_sigma_list_avg.append([timestamp,mean_avg,sigma_avg,np.abs(fit_sigma_avg)])

            last_avg = avg_booster_meas_T #just before the next iteration, last_avg becomes the avg we computed on the current iteration to compute the next running avg
        else:
            print('0.0 file encountered on file '+str(timestamp))
            zero_files_counter +=1
            #send_alert('Last file saved by DAQ is a 0.0 file. Please investigate.')
    
    avg_data_dataframe = pd.DataFrame(avg_booster_meas_T)
    #avg_data_dataframe.set_index('Frequency (Hz)',inplace=True)
    avg_data_dataframe.to_csv('monitoring_data/avg_data_dataframe.csv', sep='\t')

    T_LNA_list,T_booster_list,T_files,time_Tsensors = get_time_days(parameter='Temperature',folder = 'C:/Madmax/MORPURGO/Temperature/*.dat')

    #---------------------------

    #----------gaussmeter----------

    B_gaussmeter,time_gaussmeter = get_time_days(parameter='B field',folder='C:/Madmax/MORPURGO/Gauss/Lakeshore/*.dat')
    mask_Bfield = np.asarray(B_gaussmeter)<1.5
    mask_Bfield_tolist = mask_Bfield.tolist()
    rampup = mask_Bfield_tolist.index(False)
    number_0field = np.sum(mask_Bfield[rampup:])
    if B_gaussmeter[-1]<1.5:
        send_alert("the last value for the B field at Booster is less than 1.5T. Please investigate")
    #-------------------------------
    save_data()
    #----------------------------------Now, a permanent loop is created to add the new monitoring plots---------------------------------------------------------------


    total_files_raw_before = new_datasets_raw_total

    MADMON_plot()
    end = time.time()
    print('Monitored '+str(file_counter)+' files in '+str(np.round(end-start,2))+' seconds')

    while True:
        total_files_raw_now = glob.glob(common_path+longtime_DAQ_path+measurement_tracked+"/rawfft/*.smp")
        if new_data_from_DAQ_counter>3:
            #no new data in the last 20 minutes, send alert
            send_alert("No new data has been stored in the last 20 minutes. Please investigate")

        if (len(total_files_raw_before) == len(total_files_raw_now)):
            new_data_from_DAQ_counter=+1
            userText, timedOut = timedInput('Type YES to finish data monitoring: ',timeout=360)
            if userText == 'YES':
                print('Saving data and finalizing code')
                #saving booster peak evolution for raw datafiles
                save_data()
                #--------------------------------------------------
                time.sleep(2)
                print('Data saved, code finalized :D')
                exit(0)

        else:
            noncorrupt_file_counter+=1
            T_LNA_list = []
            T_booster_list = []
            new_data_from_DAQ_counter =0 #only for alert system
            file_counter += 1 #to count total number of files
            print('New data saved from DAQ, total now = '+str(file_counter))
            sample_counter+=1 #a new datafile has been added to the saved data folders
            #read the last 4 files written
            new_datasets_raw = sorted(total_files_raw_now)[-4:] #sorted filenames without path, taking the last 4
            timestamp = int(os.path.basename(new_datasets_raw[0])[6:-9])
            timestamp_list.append(timestamp)
            #convert the y axis to noise temperature and obtain the mean noise T per bin
            raw_booster_meas_T,time_noiseT = Power_to_noiseT(new_datasets_raw,timestamp),get_time_days(parameter='Noise T',folder = timestamp_list)
            if any(raw_booster_meas_T.index != 0)and(any(raw_booster_meas_T.values != 0)): #only proceed if this is not a 0 file
                raw_booster_meas_T_perBin = avg_noise_bins(raw_booster_meas_T)
        
                avg_booster_meas_T = running_avg(raw_booster_meas_T,noncorrupt_file_counter,last_avg)
                avg_booster_meas_T_perBin = avg_noise_bins(avg_booster_meas_T)

                middle_point_raw =int(np.round(len(raw_booster_meas_T.index)/2,0))
                mean_boosterPeak_raw=raw_booster_meas_T.iloc[middle_point_raw-10:middle_point_raw+10].mean()

                middle_point_avg =int(np.round(len(avg_booster_meas_T.index)/2,0))
                mean_boosterPeak_avg=avg_booster_meas_T.iloc[middle_point_avg-10:middle_point_avg+10].mean()
                
                baseline_savgol_raw = savgol_filter(raw_booster_meas_T['Temperature (K)'],filter_window_raw,2,mode='constant')[exclude_points_raw:-exclude_points_raw]
                baseline_savgol_avg = savgol_filter(avg_booster_meas_T['Temperature (K)'],filter_window_avg,2,mode='constant')[exclude_points_avg:-exclude_points_avg]

                #we remove the baseline by dividing the original dataset to the sav golay smooth
                booster_no_baseline_raw = pd.DataFrame((-1 + raw_booster_meas_T['Temperature (K)'].iloc[exclude_points_raw:-exclude_points_raw]/baseline_savgol_raw))
                booster_no_baseline_avg = pd.DataFrame((-1 + avg_booster_meas_T['Temperature (K)'].iloc[exclude_points_avg:-exclude_points_avg]/baseline_savgol_avg))
            
                #creating a histogram of the y values to check gaussianity
                try:
                    hist_values_raw, hist_binEdges_raw = np.histogram(booster_no_baseline_raw['Temperature (K)'],bins='auto')
                    hist_values_avg, hist_binEdges_avg = np.histogram(booster_no_baseline_avg['Temperature (K)'],bins='auto')
                except:
                    print('Could not compute histograms on file '+ str(sample_counter))

                #name of the files to use as labels and to save the figures, both variables should actually match
                data_label_raw = os.path.basename(os.path.commonprefix(new_datasets_raw))[:-3] #path basename takes only the filename, not the address. common prefix takes the common name, [:-3] removes _S0
                

                #computing mean and std as initial guess for the gaussian fit
                mean_raw = booster_no_baseline_raw['Temperature (K)'].mean() #initial guess for the fit
                mean_avg = booster_no_baseline_avg['Temperature (K)'].mean()
                                
                sigma_raw = booster_no_baseline_raw['Temperature (K)'].std() #initial guess for the fit
                
                sigma_avg = booster_no_baseline_avg['Temperature (K)'].std()
                

                #bin centers to plot gaussian fit
                bin_centers_raw = 0.5*(hist_binEdges_raw[1:] + hist_binEdges_raw[:-1])
                bin_centers_avg = 0.5*(hist_binEdges_avg[1:] + hist_binEdges_avg[:-1])
                
                try:
                    #doing a gaussian fit
                    popt_raw,pcov_raw = curve_fit(gaus,bin_centers_raw,hist_values_raw,p0=[1,mean_raw,sigma_raw])
                    popt_avg,pcov_avg = curve_fit(gaus,bin_centers_avg,hist_values_avg,p0=[1,mean_avg,sigma_avg])


                    residuals_raw = hist_values_raw - gaus(bin_centers_raw, *popt_raw)
                    ss_res_raw = np.sum(residuals_raw**2)
                    ss_tot_raw = np.sum((hist_values_raw-np.mean(hist_values_raw))**2)
                    r2_raw = 1 - (ss_res_raw / ss_tot_raw)

                    residuals_avg = hist_values_avg - gaus(bin_centers_avg, *popt_avg)
                    ss_res_avg = np.sum(residuals_avg**2)
                    ss_tot_avg = np.sum((hist_values_avg-np.mean(hist_values_avg))**2)
                    r2_avg = 1 - (ss_res_avg / ss_tot_avg)

                    r2_raw_list.append(r2_raw)
                    r2_avg_list.append(r2_avg)

                except:
                    print('Could not fit gaussian')

                sigma_list_raw.append(sigma_raw)
                mean_sigma_list_raw.append([timestamp,mean_raw,sigma_raw])
                sigma_list_avg.append(sigma_avg)
                mean_sigma_list_avg.append([timestamp,mean_raw,sigma_avg])
                TperBin_list_raw.append(np.insert(raw_booster_meas_T_perBin['Temperature (K)'].values,0,timestamp,axis=0))
                TperBin_list_avg.append(np.insert(avg_booster_meas_T_perBin['Temperature (K)'].values,0,timestamp,axis=0)) #where to insert, index to insert, value to insert, axis
                boosterPeak_T_evolution_raw.append([timestamp,float(mean_boosterPeak_raw.values)])
                boosterPeak_T_evolution_avg.append([timestamp,float(mean_boosterPeak_avg.values)])

                B_gaussmeter,time_gaussmeter = get_time_days(parameter='B field',folder='C:/Madmax/MORPURGO/Gauss/Lakeshore/*.dat')
                mask_Bfield = np.asarray(B_gaussmeter)<1.5
                mask_Bfield_tolist = mask_Bfield.tolist()
                rampup = mask_Bfield_tolist.index(False)
                number_0field = np.sum(mask_Bfield[rampup:]) #starting after initial rampup, how many Trues we have in the negation of the mask, i.e., when Bgaussmeter>1.5
                if B_gaussmeter[-1]<1.5:
                    send_alert("the last value for the B field at Booster is less than 1.5T. Please investigate")

                T_LNA_list,T_booster_list,T_files,time_Tsensors =  get_time_days(parameter='Temperature',folder = 'C:/Madmax/MORPURGO/Temperature/*.dat')

                if T_LNA_list[-1]>25:
                    send_alert("The last LNA temperature reading is above 25 degrees Celsius. Please investigate")
                if np.asarray(T_booster_list)[-1]>25:
                    send_alert("The last Spectrum Analyzer temperature reading is above 25 degrees Celsius. Please investigate")

                #Plotting---------------------------------------
                MADMON_plot()
                #End of plotting--------------------------------
                last_avg = avg_booster_meas_T #for the next running average
                #save average dataframe
                avg_data_dataframe = pd.DataFrame(avg_booster_meas_T)
                avg_data_dataframe.to_csv('monitoring_data/avg_data_dataframe.csv', sep='\t')
            else:
                print('0.0 file encountered on '+str(sample_counter))
                send_alert('Could not compute histogram on last datafiles. This usually happens when a 0.0 file is stored')
                zero_files_counter+=1
            

        if (sample_counter%10==0) and (sample_counter!=0):
            #checkpoint: saving and updating data monitoring files every 10 monitored measurements
            save_data()

        total_files_raw_before = total_files_raw_now

        time.sleep(2) 