'''                           #######################
#--------------------------   ## ---   Set up  --- ##     --------------------
                              #######################
'''
import os
import librosa
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

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
    

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




def parse_audio:
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
    
def plot_specgram(sound_names, raw_sounds, graph_folder):
    i = 1
    # fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        fig = plt.figure()
        specgram(np.array(f), Fs=22050)
        # plt.title(n.title())
        i += 1
        plt.suptitle(n,fontsize=18)
        # plt.show()
        fig.savefig(n + ".png")
        plt.close()



'''                              #######################
#-----------------------------   ## ---    MAIN   --- ##   -------------------
                                 #######################
'''

if __name__ == '__main__':
    path = os.getcwd()
    # folder = open_folder()  
    folder = path
    wav_names = get_file_names(folder)
    



raw_sounds = load_sound_files(wav_names)

graph_folder = path+'/graphs'

if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)
    
plot_specgram(wav_names,raw_sounds)


