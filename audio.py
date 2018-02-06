#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:43:11 2017

@author: BhavikaJalli
"""
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
import os
import librosa
import librosa.display

audio_path = './data/train_data/BassClarinet_01.wav'
y, sr = librosa.load(audio_path)

S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.logamplitude(S, ref_power=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('mel power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.show()
