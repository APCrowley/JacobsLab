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




'''                            #######################
#---------------------------   ## --- Functions --- ##     ---------------------
                               #######################
'''

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



'''                              #######################
#-----------------------------   ## ---    MAIN   --- ##   -------------------
                                 #######################
'''

if __name__ == '__main__':
    path = os.getcwd()
    folder = open_folder()  
    wav_names = get_file_names(folder)
    
def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_specgram(sound_names,raw_sounds):
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

raw_sounds = load_sound_files(wav_names)

plot_specgram(wav_names,raw_sounds)


# test
a=8
    
    