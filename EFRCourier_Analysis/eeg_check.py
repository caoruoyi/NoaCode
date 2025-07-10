import numpy as np
from ptsa.data.filters import ResampleFilter, ButterworthFilter, MorletWaveletFilter
import cmlreaders as cml
import scipy as sp
import xarray as xr
import cc_utils as cc
from scipy import fft
from scipy.stats import linregress
from adjustText import adjust_text
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def eeg_check(subject, session, exp):
#     try:
        pow_params = cc.pow_params
        period = 'encoding' # 'retrieval' 'encoding'
        POWHZ = 50. # downsampling factor
        FREQS = np.logspace(np.log10(3),np.log10(180), 8)

        select_subs = 1
        test_subjects = [subject] # **select subject here**

        df = cml.CMLReader.get_data_index()
        exp_df = df[(df.subject == subject) & (df.experiment == exp) & (df.session == session)]
        del df

        if len(exp_df) != 1:
            raise KeyError('The wrong number of sessions inputted' + str(l))

        reader = cml.CMLReader(subject=exp_df['subject'].iloc[0], 
                               experiment=exp_df['experiment'].iloc[0], session=exp_df['session'].iloc[0], 
                               localization=exp_df['localization'].iloc[0], montage=exp_df['montage'].iloc[0])
        events = reader.load('task_events')
        if exp == 'RepFR2':
            events = events.drop(columns=['stim_params']) # otherwise get unhashable dict error from xarray
        ev_type = pow_params[exp][period]['ev_type']
        events = events.query('type == @ev_type')
        if period == 'encoding':
            events = cc.process_events(events)
        elif period == 'retrieval':
            events = cc.process_events_retrieval(events) 
        elif period == 'countdown':
            events = cc.process_events_countdown(events) 

        scheme = reader.load("pairs")
        buf = pow_params[exp][period]['buf'] * 1000
        rel_start = pow_params[exp][period]['start_time'] * 1000
        rel_stop = pow_params[exp][period]['end_time'] * 1000

        if pow_params[exp][period]['mirror']: # use mirror for retrieval for unknown reasons
            dat = reader.load_eeg(events=events,
                              rel_start=rel_start,
                              rel_stop=rel_stop,
                              scheme=scheme).to_ptsa()
            dat['time'] = dat['time'] / 1000 # using PTSA time scale
            dat = dat.add_mirror_buffer(pow_params[exp][period]['buf'])
            dat['time'] = dat['time'] * 1000
        else:
            dat = reader.load_eeg(events=events,
                              rel_start=-buf+rel_start,
                              rel_stop=buf+rel_stop,
                              scheme=scheme).to_ptsa()
        del reader; del events; del pow_params;

        dat = dat.astype(float) - dat.mean('time')
        
        
        # do each step twice with split channels
        half_elec = int(np.shape(dat)[1]/2)
        
        sr = dat.samplerate.values
        # Notch Butterworth filter for 60Hz line noise:
        dat1 = ButterworthFilter(freq_range=[58, 62], filt_type='stop', order=4).filter(timeseries=dat[:,:half_elec,:])
        dat2 = ButterworthFilter(freq_range=[58, 62], filt_type='stop', order=4).filter(timeseries=dat[:,half_elec:,:])
        
        dat1 = MorletWaveletFilter(freqs=FREQS, output='power',width=5, verbose=True).filter(timeseries=dat1)
        dat2 = MorletWaveletFilter(freqs=FREQS, output='power',width=5, verbose=True).filter(timeseries=dat2)

        dat1 = xr.ufuncs.log10(dat1, out=dat1.values)
        dat2 = xr.ufuncs.log10(dat2, out=dat2.values)

        dat1 = ResampleFilter(resamplerate=POWHZ).filter(timeseries=dat1) # 8 x trials x elecs x 4594 resampled to 20 ms bins
        dat2 = ResampleFilter(resamplerate=POWHZ).filter(timeseries=dat2)
        #     print('done resample')

         ## while dat is still loaded save the original line_filt voltages
        filter_noise_lines = 1

        if filter_noise_lines == 1:
            # line noise removal
            line_filt_dat = ButterworthFilter(freq_range=[58., 62.], filt_type='stop',order=4).filter(timeseries=dat)
            line_filt_dat = ButterworthFilter(freq_range=[118., 122.], filt_type='stop',order=4).filter(timeseries=line_filt_dat)        
            line_filt_dat = ButterworthFilter(freq_range=[178., 182.], filt_type='stop',order=4).filter(timeseries=line_filt_dat)                
        else:
            line_filt_dat = dat     
        del dat;

        dat = dat1.append(dat2,'channel') 
        del dat1; del dat2;

        dat = dat.remove_buffer(duration=(buf / 1000))

        dat = dat.mean('time') # average over time so 8 x trials x elecs

        if (period == 'encoding') | (period == 'countdown') | (period == 'stim_on'):
            z_pow_all = xr.apply_ufunc(sp.stats.zscore, dat, 
                                       input_core_dims=[['channel', 'frequency']], 
                                       output_core_dims=[['channel', 'frequency']]) # outputs trials x elecs x 8
        elif period == 'retrieval':
            z_pow_all = xr.apply_ufunc(sp.stats.zscore, dat, 
                                       input_core_dims=[['channel', 'frequency', 'time']], 
                                       output_core_dims=[['channel', 'frequency', 'time']])
        del dat

        N = 1000 #np.shape(line_filt_dat)[2] # Number of samplepoints 
        T = 1.0 / N # sample spacing 
        sr = line_filt_dat.samplerate.values

        time_range = range(0,int(2000*(sr/1000))) # 2000 ms
        plot_range = range(5,382) # first few are strange...then only go to 200 Hz (which is 400)
        shift_factor = 1 # taking different time chunks shifts the frequencies
        
        ch_z_pow = []
        ch_std = []

        for ch in range(np.shape(line_filt_dat)[1]):

            z_pows = []
            temp_std = []

            for tr in range(np.shape(line_filt_dat)[0]):

                y = line_filt_dat[tr,ch,time_range]

                temp_std.append(np.std(y))
                if tr < np.shape(line_filt_dat)[0]-1: # last one don't look for correlation
                    z_pows.append(np.corrcoef(z_pow_all[tr,ch,:],z_pow_all[tr+1,ch,:])[0,1])

            ch_z_pow.append(np.mean(z_pows))
            ch_std.append(np.mean(temp_std))

        # values out of range to min and max values for visualization
        max_corr = 0.6
        min_corr = 0
        max_std = 4e4
        min_std = 10

        temp_ch_z_pow = np.array(ch_z_pow); 
        temp_ch_z_pow[temp_ch_z_pow>max_corr] = max_corr
        temp_ch_z_pow[temp_ch_z_pow<min_corr] = min_corr

        temp_ch_std = np.array(ch_std)
        temp_ch_std[temp_ch_std>max_std] = max_std
        temp_ch_std[temp_ch_std<min_std] = min_std
        
        return temp_ch_std, temp_ch_z_pow, max_corr, min_corr, max_std, min_std
#     except:
#         return subject+'_'+str(session)+' failed'
    

def plot_eeg_check(check_results, subject, session, sess_i, pairs):
    
    temp_ch_std, temp_ch_z_pow, max_corr, min_corr, max_std, min_std = check_results[sess_i]
    
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":12,"axes.labelsize":12,
                                "axes.ticksize":12,"axes.titlesize":12})

    def scatterWithFirstSecondStds(ax,data1,data2,improve_labels = False):
        ax.scatter(data1, data2)
#         This code shows electrode NUMBERS
#         texts = [plt.text(data1[i], data2[i],txt,ha='left', va='bottom') for i, txt in enumerate(range(len(data1)))]

        # This code shows electrode LABELS
        texts = [plt.text(data1[i], data2[i],txt,ha='left', va='bottom') for i, txt in enumerate(pairs)]  
#         adjust_text(texts) # this is cool but it takes dozens of seconds
        ax.vlines(np.mean(data1)+np.std(data1),np.min(data2),np.max(data2),color=(0.8,0.8,1),linestyle='--') # 1 std
        ax.hlines(np.mean(data2)+np.std(data2),np.min(data1),np.max(data1),color=(0.8,0.8,1),linestyle='--') # 1 std
        ax.vlines(np.mean(data1)+2*np.std(data1),np.min(data2),np.max(data2),color=(1,0.9,0.9),linestyle='--') # 2 std
        ax.hlines(np.mean(data2)+2*np.std(data2),np.min(data1),np.max(data1),color=(1,0.9,0.9),linestyle='--') # 2 std
        return ax,texts

    # plot correlations of spectral powers at arbitrary time points vs. std of raw values
    plt.subplots(1,1,figsize=(10,7))
    ax = plt.subplot(1,1,1)
    ax,texts = scatterWithFirstSecondStds(ax,temp_ch_std, temp_ch_z_pow)
    plt.xlabel('Standard deviation of raw timeseries')
    plt.ylabel('Correlation of spectral features trial N to N+1')
    ax.set_xscale('log')
    ax.set_xlim(min_std,max_std)
    ax.set_ylim(min_corr-0.02,max_corr+0.02)
    # adjust_text(texts)
    ax.set_title(subject+', session '+str(session))
    
def plot_eeg_check_jlab(subject, session, exp, pairs):
    
    temp_ch_std, temp_ch_z_pow, max_corr, min_corr, max_std, min_std = eeg_check(subject, session, exp)
    
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":12,"axes.labelsize":12,
                                "axes.ticksize":12,"axes.titlesize":12})

    def scatterWithFirstSecondStds(ax,data1,data2,improve_labels = False):
        ax.scatter(data1, data2)
#         This code shows electrode NUMBERS
#         texts = [plt.text(data1[i], data2[i],txt,ha='left', va='bottom') for i, txt in enumerate(range(len(data1)))]

        # This code shows electrode LABELS
        texts = [plt.text(data1[i], data2[i],txt,ha='left', va='bottom') for i, txt in enumerate(pairs)]  
#         adjust_text(texts) # this is cool but it takes dozens of seconds
        ax.vlines(np.mean(data1)+np.std(data1),np.min(data2),np.max(data2),color=(0.8,0.8,1),linestyle='--') # 1 std
        ax.hlines(np.mean(data2)+np.std(data2),np.min(data1),np.max(data1),color=(0.8,0.8,1),linestyle='--') # 1 std
        ax.vlines(np.mean(data1)+2*np.std(data1),np.min(data2),np.max(data2),color=(1,0.9,0.9),linestyle='--') # 2 std
        ax.hlines(np.mean(data2)+2*np.std(data2),np.min(data1),np.max(data1),color=(1,0.9,0.9),linestyle='--') # 2 std
        return ax,texts

    # plot correlations of spectral powers at arbitrary time points vs. std of raw values
    plt.subplots(1,1,figsize=(10,7))
    ax = plt.subplot(1,1,1)
    ax,texts = scatterWithFirstSecondStds(ax,temp_ch_std, temp_ch_z_pow)
    plt.xlabel('Standard deviation of raw timeseries')
    plt.ylabel('Correlation of spectral features trial N to N+1')
    ax.set_xscale('log')
    ax.set_xlim(min_std,max_std)
    ax.set_ylim(min_corr-0.02,max_corr+0.02)
    # adjust_text(texts)
    ax.set_title(subject+', session '+str(session))    
    
    
def erp_sme(subject, session):
    print('Session ' + str(session))
    data = cml.get_data_index(kind = 'r1'); data = data[data['experiment'] == 'RepFR1']; data = data[data['subject'] == subject]
    loc = data[data.session == session].localization.iloc[0]
    mon = data[data.session == session].montage.iloc[0]
    r = cml.CMLReader(subject=subject, experiment='RepFR1', session=session, localization=loc, montage=mon)
    evs = r.load('task_events')

    word_evs = evs[evs.type=='WORD']
    rec_evs = evs[evs.type=='REC_WORD']

    pairs = r.load('pairs')

    word_evs['eegfile'].unique()

    contacts = r.load('contacts')

    buf = 500
    eeg = r.load_eeg(word_evs, rel_start=-500 - buf, rel_stop=500 + buf, scheme=contacts)

    eeg_ptsa = eeg.to_ptsa()
    # Butterworth filter to remove 60 Hz line noise + harmonic at 120
    eeg_ptsa = eeg_ptsa.filtered([58, 62])
    eeg_ptsa = eeg_ptsa.filtered([118, 122])

    from ptsa.data.filters import MorletWaveletFilter

    erp = eeg_ptsa[word_evs.recalled.astype(bool).values].mean(['event', 'channel']) - eeg_ptsa[~word_evs.recalled.astype(bool).values].mean(['event', 'channel'])

    plt.figure(figsize=(8,6))
    e = erp.plot()

    pows = MorletWaveletFilter(freqs=np.logspace(np.log10(6), np.log10(180), 8), output='power', width=4, cpus=10).filter(timeseries=eeg_ptsa)

    pows = pows.remove_buffer(0.5)

    mask = word_evs.recalled.astype(bool).values
    sme = pows[:, mask].mean(['event', 'channel']) - pows[:, ~mask].mean(['event', 'channel'])

    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    im = ax.imshow(sme, cmap='coolwarm', aspect = 20, interpolation='hamming')
    plt.yticks(ticks = range(0, 8), labels=[f"{f:.0f}" for f in np.logspace(np.log10(6), np.log10(180), 8)])
    labels = ax.get_xticklabels()
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Frequencies', fontsize=20)
    plt.title('Subsequent Memory Effect', fontsize=24)
    cbar = plt.colorbar(im, ax=ax, fraction = .01)
    cbar.set_label(label='t statistic \n (remembered - forgotten)', size=16)
    cbar.ax.tick_params(labelsize=14)
    plt.show()