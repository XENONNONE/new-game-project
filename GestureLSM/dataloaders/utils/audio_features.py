"""modified from https://github.com/yesheng-THU/GFGE/blob/main/data_processing/audio_features.py"""
import numpy as np
import librosa
import math
import os
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
from typing import Optional, Tuple
from numpy.lib import stride_tricks
from loguru import logger



def process_audio_data(audio_file, args, data, f_name, selected_file):
    """Process audio data with support for different representations."""
    logger.info(f"# ---- Building cache for Audio {f_name} ---- #")
    
    if not os.path.exists(audio_file):
        logger.warning(f"# ---- file not found for Audio {f_name}, skip all files with the same id ---- #")
        selected_file.drop(selected_file[selected_file['id'] == f_name].index, inplace=True)
        return None

    audio_save_path = audio_file.replace("wave16k", "onset_amplitude").replace(".wav", ".npy")
    
    if args.audio_rep == "onset+amplitude" and os.path.exists(audio_save_path):
        data['audio'] = np.load(audio_save_path)
        logger.warning(f"# ---- file found cache for Audio {f_name} ---- #")
    
    elif args.audio_rep == "onset+amplitude":
        data['audio'] = calculate_onset_amplitude(audio_file, args.audio_sr, audio_save_path)
        
    elif args.audio_rep == "mfcc":
        audio_data, _ = librosa.load(audio_file)
        data['audio'] = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=args.audio_sr, 
            n_mels=128, 
            hop_length=int(args.audio_sr/args.audio_fps)
        ).transpose(1, 0)
    
    if args.audio_norm and args.audio_rep == "wave16k":
        data['audio'] = (data['audio'] - args.mean_audio) / args.std_audio
    
    return data

def calculate_onset_amplitude(audio_file, audio_sr, save_path):
    """Calculate onset and amplitude features from audio file."""
    audio_data, sr = librosa.load(audio_file)
    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=audio_sr)
    
    # Calculate amplitude envelope
    frame_length = 1024
    shape = (audio_data.shape[-1] - frame_length + 1, frame_length)
    strides = (audio_data.strides[-1], audio_data.strides[-1])
    rolling_view = stride_tricks.as_strided(audio_data, shape=shape, strides=strides)
    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
    
    # Calculate onset
    audio_onset_f = librosa.onset.onset_detect(y=audio_data, sr=audio_sr, units='frames')
    onset_array = np.zeros(len(audio_data), dtype=float)
    onset_array[audio_onset_f] = 1.0
    
    # Combine features
    features = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
    
    # Save features
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, features)
    
    return features