import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib

from transformers import AutoFeatureExtractor, ClapModel
import torch
import torchaudio

import os
from os.path import join as opj
from os.path import join, exists, split
import time
import urllib.request
import warnings
from tqdm import tqdm
from pprint import pprint
import zipfile
import glob

from glmsingle.glmsingle import GLM_single
import pandas as pd
from nilearn import maskers
from nilearn import plotting
import tqdm

from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs, mean_img
import matplotlib.pyplot as plt
import nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_contrast_matrix

TR=1.5
stimdur=15
how_many_fmri_vols=10
fmri_vol_delay=3

def load_data(subj, avg=True, mask=False):
    base_data_path="/data01/data/fMRI_music_genre/ds003720-download/derivatives"
    base_event_path="/data01/data/fMRI_music_genre/ds003720-download"

    data_path=opj(base_data_path,subj,"func")
    # event_path=opj(base_data_path,"events")

    stim_dir="/data01/data/fMRI_music_genre/data_wav/genres_original"

    sessions=os.listdir(data_path)
    filenames=glob.glob(opj(data_path,"*-preproc_bold.nii.gz"))

    data=[]
    events=[]
    drop_first=0

    for fn in tqdm.tqdm(filenames):
        #load file
        print(data_path,fn)
        
        name=fn.split("/")[-1].replace("_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz","_events.tsv")
        
        events_path=opj(base_event_path,subj,"func",name)
        # print(opj(data_path_ses,fn),opj(event_path,fn.replace('_bold_preproc.nii.gz','_events.tsv')))

        x=nib.load(fn)
        # df=pd.read_csv(fn.replace('_bold.nii','_events.tsv'),sep='\t',)
        df=pd.read_csv(events_path,sep='\t',)
        
        events.append(df)
        data.append(x)


    masker = maskers.NiftiMasker(mask_strategy="epi")
    masker.fit(data[0])
    report = masker.generate_report()

    # tgt_dir=f"preprocessed_data/{subj}"
    tgt_dir=f"/data01/data/fMRI_music_genre/fmri_preproc_data/{subj}"

    if mask:
        print("Masking files..")

        for i in tqdm.trange(len(data)):
            data[i]=masker.transform_single_imgs(data[i]).T
        
        os.makedirs(tgt_dir,exist_ok=True)

        for i in range(len(data)):
            print(data[i].shape)
            np.save(f"{tgt_dir}/{filenames[i].split('/')[-1].replace('.nii.gz','.npy')}",data[i])
    else:
        for i,f in enumerate(filenames):
            data[i]=np.load(f"{tgt_dir}/{f.split('/')[-1].replace('.nii.gz','.npy')}")

            
    for e in events:
        # e["category"]=e["genre"]+e["track"]
        e['category'] = e.apply(lambda row: f"{row['genre']}_{row['track']}", axis=1)


        e["trial_type"]=e["category"]


    design = []
    conditions=[]

    print("Loading nifti and events..")
    for i in tqdm.trange(len(data)):
            run_events = events[i]

            #compute number of conds
            conds=run_events["category"].unique().tolist()
            nconds = len(conds)

            run_design = np.zeros((np.shape(data[i])[-1], nconds))
            for c, cond in enumerate(conds):
                    condidx = np.argwhere(run_events['category'].values == cond)
                    
                    # condvols = (run_events.onset.values[condidx]//TR).astype(np.int64).squeeze()
                    start_condvols=(run_events.onset.values[condidx]//TR).astype(np.int64)
                    indices=[]
                    condvols = np.array([np.arange(start_idx,start_idx+int(stimdur//TR)) for start_idx in start_condvols])
                    
                    run_design[condvols, c] = 1

            conditions.append(conds)
            design.append(run_design)
            # data[i]=data[i]
    print("Detrending fMRI data..")

    cleaned_data=[nilearn.signal.clean(d.T,detrend=True,standardize=True,t_r=TR) for d in data] # detrend and standardize

    print("Preparing dataset..")

    train_fmri=[]
    test_fmri=[]
    train_audio=[]
    test_audio=[]
    
    train_stim_name=[]
    test_stim_name=[]

    train_sr=[]
    test_sr=[]

    train_genre=[]
    test_genre=[]

    for run_data, run_event,fn in zip(cleaned_data, events,filenames):
        print(fn)

        for i in range(len(run_event)):
            start_vol=(run_event.iloc[i].onset//TR).astype(np.int64) + fmri_vol_delay

            tmp_fmri_data=run_data[start_vol:start_vol+how_many_fmri_vols]
            tmp_stim=opj(stim_dir,run_event.iloc[i].genre.replace("'",""),run_event.iloc[i].genre.replace("'","")+"."+f"{run_event.iloc[i].track}".zfill(5)+".wav")
            audio,sr=torchaudio.load(tmp_stim)

            if "task-Test" in fn:
                test_fmri.append(tmp_fmri_data)
                test_audio.append(audio)
                test_sr.append(sr)
                test_genre.append(run_event.iloc[i].genre.replace("'",""))
                test_stim_name.append(tmp_stim)

            elif "task-Training" in fn:
                train_fmri.append(tmp_fmri_data)
                train_audio.append(audio)
                train_sr.append(sr)
                train_genre.append(run_event.iloc[i].genre.replace("'",""))
                train_stim_name.append(tmp_stim)

            else:
                print("ei")

    if avg:
        train_fmri_avg=[]
        test_fmri_avg=[]
        for f in tqdm.tqdm(train_fmri):
            train_fmri_avg.append(f.mean(0))
        for f in tqdm.tqdm(test_fmri):
            test_fmri_avg.append(f.mean(0))
        train_fmri_avg=torch.tensor(train_fmri_avg)
        test_fmri_avg=torch.tensor(test_fmri_avg)
    else: 
        train_fmri_avg=[]
        test_fmri_avg=[]
        for f in tqdm.tqdm(train_fmri):
            if (f.shape[0]<how_many_fmri_vols):
                pad_width = ((0, how_many_fmri_vols - f.shape[0]), (0, 0))
                f = np.pad(f, pad_width, mode='edge')
            train_fmri_avg.append(torch.tensor(f))
        for f in tqdm.tqdm(test_fmri):
            if (f.shape[0]<how_many_fmri_vols):
                pad_width = ((0, how_many_fmri_vols - f.shape[0]), (0, 0))
                f = np.pad(f, pad_width, mode='edge')
            test_fmri_avg.append(torch.tensor(f))
        train_fmri_avg = torch.stack(train_fmri_avg)
        test_fmri_avg = torch.stack(test_fmri_avg)
    
    #"laion/clap-htsat-unfused"
    clap_model_id="laion/larger_clap_music_and_speech"
    model = ClapModel.from_pretrained(clap_model_id).to("cuda")
    feature_extractor = AutoFeatureExtractor.from_pretrained(clap_model_id)

    train_audio_feat=[]
    test_audio_feat=[]

    print("Audio feature extraction...")

    resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=48000)

    with torch.no_grad():
        for wv,sr in tqdm.tqdm(zip(train_audio,train_sr)):
            wv = resampler(wv)
            inputs = feature_extractor(wv.squeeze(), return_tensors="pt",sampling_rate=48_000)
            audio_features = model.get_audio_features(inputs.input_features.to("cuda")).cpu()
            train_audio_feat.append(audio_features)

    with torch.no_grad():
        for wv,sr in tqdm.tqdm(zip(test_audio,train_sr)):
            wv = resampler(wv)
            inputs = feature_extractor(wv.squeeze(), return_tensors="pt",sampling_rate=48_000)
            audio_features = model.get_audio_features(inputs.input_features.to("cuda")).cpu()
            test_audio_feat.append(audio_features)


    train_audio_feat=torch.stack(train_audio_feat).squeeze()
    test_audio_feat=torch.stack(test_audio_feat).squeeze()


    return train_fmri_avg,train_audio_feat,train_genre,train_audio,test_fmri_avg,test_audio_feat,test_genre,test_audio, masker, train_stim_name, test_stim_name 
        
        