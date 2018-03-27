'''                           #######################
#--------------------------   ## ---   Set up  --- ##     --------------------
                              #######################
'''
import os
import librosa
import librosa.display
import pandas as pd
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
git commit -m 'message'
git push origin master

If the branch is not available, ensure that the remote url is 
https://github.com/AndrewPCrowley/JacobsLab.git
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
        elif '.png' in f:
           file_names.remove(f)
        elif '-' not in f:
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


def create_audio_X(wav_fullpaths, wav_names, plot='no'):
    features, labels = np.empty((0,193)), np.empty(0)
    for fp, f in zip(wav_fullpaths, wav_names):
        stft, mfccs, chroma, mel, contrast,tonnetz, raw_sound, sample_rate = extract_feature(fp)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        if '/sniff' in f:
            labels = np.append(labels, 'sniff')
        else:
            labels = np.append(labels,'other')
    
        if plot=='yes':
            plot_audio_features(raw_sound, sample_rate, f, graph_folder)
            
        
    return features, labels


        
def plot_audio_features(raw_sound, sample_rate, file_name, graph_folder):
    # Short-time Fourier transform (STFT) spectrograms
    stft_folder = graph_folder+'/STFT_spec/'
    if not os.path.exists(stft_folder):
        os.makedirs(stft_folder)

    stft_sound = np.abs(librosa.stft(raw_sound))
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(stft_sound, ref=np.max),
                             y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
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
    plt.colorbar()
    plt.title('MFCC')
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
    plt.colorbar()
    plt.title('Chromagram')
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
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    fig.savefig(mel_folder + file_name +"_mel.png")
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
        # plt.title(n.title())
        plt.suptitle(n,fontsize=18)
        # plt.show()
        fig.savefig(spectrogram_folder + n +".png")
        plt.close()



'''                              #######################
#-----------------------------   ## ---    MAIN   --- ##   -------------------
                                 #######################
'''

if __name__ == '__main__':
    path = os.getcwd()
    # folder = open_folder()  
    folder = path+'/clips'
    wav_names = get_file_names(folder)
    wav_fullpaths = [folder+'/'+name for name in wav_names]

    # # Load audio time series for all wav files
    raw_sounds = load_sound_files(wav_fullpaths)

    # # create folder to save graphs
    graph_folder = path+'/graphs'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
    
    plot_specgram(wav_names, raw_sounds, graph_folder)

    # # extract features for each wav file
    # stft, mfccs,chroma,mel,contrast,tonnetz = extract_feature(wav_fullpaths[0])
    features, labels = create_audio_X(wav_fullpaths, wav_names, plot='yes')
    





