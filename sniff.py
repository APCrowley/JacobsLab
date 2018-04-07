'''                           #######################
#--------------------------   ## ---   Set up  --- ##     --------------------
                              #######################
'''
import os
import librosa
import librosa.display
import pandas as pd
pd.set_option('display.max_columns',0)
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram
import tkinter as Tk #opening files
from tkinter import filedialog


'''                            #######################
#---------------------------   ## ---   Notes --- ##     ---------------------
                               #######################
'''

''' 

git pull origin master
git add whatever is being add (folder, file)
git add -NAME .             (add everything but NAME)
git commit -m 'message'
git push origin master

If the branch is not available, ensure that the remote url is 
https://github.com/AndrewPCrowley/JacobsLab.git

Do not add  anything but sniff clips to the github or you'll probably crash the repo
'''



'''                            #######################
#---------------------------   ## --- Functions --- ##     ---------------------
                               #######################
'''

def extract_feature(wav_path):
    X, sample_rate = librosa.load(wav_path)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return stft, mfccs,chroma,mel,contrast, tonnetz, X, sample_rate
    

def get_file_names(folder):
    # Read all the names of the files and store as list
    file_names_temp = os.listdir(folder)
    file_names = os.listdir(folder)

    for f in file_names_temp:
        if (f.startswith('.')==True):
           file_names.remove(f)
    return file_names


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def open_folder():
    ''' 
    Function: Opens a GUI to find a folder pathway
    Inputs: none
    Returns: pd dataframe
    '''
    
    root = Tk.Tk()
    
    #stop extra root window from Tk opening
    root.withdraw()

    root.update()
    
    file_path = Tk.filedialog.askdirectory()
    
    root.quit() 
    
    return file_path


def create_audio_X(sub_trial, wav_fullpaths, wav_names):
    features, labels = np.empty((0,195)), np.empty(0)
    for fp, f in zip(wav_fullpaths, wav_names):
        sub_trial = sub_trial
        clip_number = f.split('-')[1].split('.')[0]
        
        stft, mfccs, chroma, mel, contrast, tonnetz, raw_sound, sample_rate = extract_feature(fp)
        print('mfccs:', mfccs.shape)
        print('chroma:', chroma.shape)
        print('mel:', mel.shape)
        print('contrast:', contrast.shape)
        print('tonnetz:', tonnetz.shape)
        clip_name = np.array((sub_trial, clip_number))
        ext_features = np.hstack([clip_name, mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        if 'sniff' in f:
            labels = np.append(labels, 'sniff')
        else:
            labels = np.append(labels,'other')
            
        
    return np.array(features), np.array(labels, dtype=str)
        
def plot_audio_features(wav_fullpaths, wav_names):
    for fp, file_name in zip(wav_fullpaths, wav_names):
        raw_sound, sample_rate = librosa.load(fp)
        
        # Short-time Fourier transform (STFT) spectrograms
        stft_folder = graph_folder+'/STFT_spec/'
        if not os.path.exists(stft_folder):
            os.makedirs(stft_folder)

        stft_sound = np.abs(librosa.stft(raw_sound))
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(stft_sound, ref=np.max),
                                 y_axis='log', x_axis='time')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(stft_folder + file_name +"_stft.png")
        plt.close()
    
    
        # Mel-frequency cepstral coefficients (MFCC) spectrograms
        mfccs_folder = graph_folder+'/mfccs_spec/'
        if not os.path.exists(mfccs_folder):
            os.makedirs(mfccs_folder)

        mfccs = librosa.feature.mfcc(y=raw_sound, sr=sample_rate, n_mfcc=40)
    
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(mfccs_folder + file_name +"_mfccs.png")
        plt.close()
    
    
        # Chromagram 
        chroma_folder = graph_folder+'/chroma_spec/'
        if not os.path.exists(chroma_folder):
            os.makedirs(chroma_folder)

        chroma = librosa.feature.chroma_stft(S=stft_sound, sr=sample_rate)
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(chroma_folder + file_name +"_chroma.png")
        plt.close()
    

        # mel-scaled spectrogram (mel)
        mel_folder = graph_folder+'/mel_spec/'
        if not os.path.exists(mel_folder):
            os.makedirs(mel_folder)
        
        mel = librosa.feature.melspectrogram(y=raw_sound, sr=sample_rate, n_mels=128, fmax=8000)
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                                 y_axis='mel', fmax=8000,
                                 x_axis='time')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(mel_folder + file_name +"_mel.png")
        plt.close()
    
    
        # spectral contrast spectrogram (mel)
        contrast_folder = graph_folder+'/constrast_spec/'
        if not os.path.exists(contrast_folder):
            os.makedirs(contrast_folder)
        
        contrast = librosa.feature.spectral_contrast(S=stft_sound, sr=sample_rate)
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(contrast, x_axis='time')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(contrast_folder + file_name +"_contrast.png")
        plt.close()
    
    
        # tonal centroid spectrogram (mel)
        tonnetz_folder = graph_folder+'/tonnetz_spec/'
        if not os.path.exists(tonnetz_folder):
            os.makedirs(tonnetz_folder)
        
        y = librosa.effects.harmonic(raw_sound)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sample_rate)
    
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(tonnetz, y_axis='tonnetz')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(tonnetz_folder + file_name +"_tonnetz.png")
        plt.close()

def plot_specgram(sound_names, raw_sounds, graph_folder):
    # Basic spectrograms
    spectrogram_folder = graph_folder+'/spectrogram/'
    if not os.path.exists(spectrogram_folder):
        os.makedirs(spectrogram_folder)
    
    # fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        
        # Basic spectrogram
        fig = plt.figure(figsize=(10, 4))
        specgram(np.array(f), Fs=22050)
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(spectrogram_folder + n +".png")
        plt.close()



'''                              #######################
#-----------------------------   ## ---    MAIN   --- ##   -------------------
                                 #######################
'''

if __name__ == '__main__':
    path = os.getcwd()
    folder = open_folder() 
    # folder = path+'/clips/test_files'
    sub_trial_names = get_file_names(folder)
    wav_names = get_file_names(folder)
    
    all_sniff_data = np.empty((0,196))
    for sub_trial in sub_trial_names:
        sub_trial_folder = folder+'/'+sub_trial
        wav_names = get_file_names(sub_trial_folder)
        print(wav_names)
    
        wav_fullpaths = [sub_trial_folder+'/'+name for name in wav_names]

        # # Load audio time series for all wav files
        raw_sounds = load_sound_files(wav_fullpaths)

        # # create folder to save graphs
        graph_folder = path+'/graphs'
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)

        # # Plotting basic spectrograms and special ones.
        plot_specgram(wav_names, raw_sounds, graph_folder)
        plot_audio_features(wav_fullpaths, wav_names)

        # # Create features
        sub_trial_features, sub_trial_labels = create_audio_X(sub_trial, wav_fullpaths, wav_names)
        
        features_labels = np.concatenate((sub_trial_features, sub_trial_labels.reshape(len(sub_trial_labels),1)), axis=1)
        all_sniff_data = np.vstack([all_sniff_data, features_labels])
        
    all_sniff_data = pd.DataFrame(all_sniff_data)
    all_sniff_data.to_csv('tmp.csv')
    
    sub_trial_num = ['sub_trial', 'clip_num']
    mfccs_header = ['mfccs_'+str(i) for i in range(40)]
    chroma_header = ['chroma_'+str(i) for i in range(12)]
    mel_header = ['mel_'+str(i) for i in range(128)]
    contrast_header = ['contrast_'+str(i) for i in range(7)]
    tonnetz_header = ['tonnetz_'+str(i) for i in range(6)]
    label_header = ['label']
    
    header = [name for group in [sub_trial_num, mfccs_header, chroma_header, mel_header, contrast_header, tonnetz_header, label_header] for name in group]
    
    all_sniff_data.columns = header
    all_sniff_data.to_csv('sniff_feat_labels.csv', index=False)

    





