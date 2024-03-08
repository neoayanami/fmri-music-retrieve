import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib
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
warnings.filterwarnings('ignore')
# from glmsingle.glmsingle import GLM_single
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

default_n_threads = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

base_data_path="/data01/data/fMRI_music_genre/ds003720-download/derivatives"  # -download
base_event_path="/data01/data/fMRI_music_genre/ds003720-download"
subj="sub-002"
data_path=opj(base_data_path,subj,"func")
stim_dir="/data01/data/fMRI_music_genre/data_wav/genres_original"
sessions=os.listdir(data_path)
filenames=glob.glob(opj(data_path,"*-preproc_bold.nii.gz"))

data=[]
events=[]
drop_first=0
for fn in tqdm.tqdm(filenames):
    #load file
    print(fn)
    name=fn.split("/")[-1].replace("_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz","_events.tsv")
    events_path=opj(base_event_path,subj,"func",name)
    x=nib.load(fn)
    df=pd.read_csv(events_path,sep='\t',)
    events.append(df)
    data.append(x)

masker = maskers.NiftiMasker(mask_strategy="epi")
masker.fit(data[0])
report = masker.generate_report()

mask=False
tgt_dir=f"/data01/data/fMRI_music_genre/fmri_preproc_data/{subj}"
if mask:
    for i in tqdm.trange(len(data)):
        data[i]=masker.transform_single_imgs(data[i]).T
    os.makedirs(tgt_dir,exist_ok=True)
    for i in range(len(data)):
        print(data[i].shape)
        print(f"{tgt_dir}/{filenames[i].split('/')[-1].replace('.nii.gz','.npy')}")
        np.save(f"{tgt_dir}/{filenames[i].split('/')[-1].replace('.nii.gz','.npy')}",data[i])
else:
    for i,f in enumerate(filenames):
        data[i]=np.load(f"{tgt_dir}/{f.split('/')[-1].replace('.nii.gz','.npy')}")

for e in events:
    e['category'] = e.apply(lambda row: f"{row['genre']}_{row['track']}", axis=1)    
    e["trial_type"]=e["category"]

TR=1.5
cleaned_data=[nilearn.signal.clean(d.T,detrend=True,standardize=True,t_r=TR) for d in data] # detrend and standardize

train_fmri=[]
test_fmri=[]
train_audio=[]
test_audio=[]
train_sr=[]
test_sr=[]

train_genre=[]
test_genre=[]

TR=1.5
stimdur=10
how_many_fmri_vols=10
fmri_vol_delay=3

for run_data, run_event, fn in zip(cleaned_data, events, filenames):
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

        elif "task-Training" in fn:
            train_fmri.append(tmp_fmri_data)
            train_audio.append(audio)
            train_sr.append(sr)
            train_genre.append(run_event.iloc[i].genre.replace("'",""))

train_fmri_avg=[]
test_fmri_avg=[]
for f in tqdm.tqdm(train_fmri):
    train_fmri_avg.append(f.mean(0))
for f in tqdm.tqdm(test_fmri):
    test_fmri_avg.append(f.mean(0))
train_fmri_avg=torch.tensor(train_fmri_avg)
test_fmri_avg=torch.tensor(test_fmri_avg)

train_genre=np.array(train_genre)
test_genre=np.array(test_genre)

from transformers import AutoFeatureExtractor, ClapModel
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
random_audio = torch.rand((16_000))
inputs = feature_extractor(random_audio, return_tensors="pt")
audio_features = model.get_audio_features(**inputs)

train_audio_feat=[]
test_audio_feat=[]
with torch.no_grad():
    for wv,sr in tqdm.tqdm(zip(train_audio,train_sr)):
        inputs = feature_extractor(wv.squeeze(), return_tensors="pt",sampling_rate=48_000)
        audio_features = model.get_audio_features(**inputs)
        train_audio_feat.append(audio_features)
with torch.no_grad():
    for wv,sr in tqdm.tqdm(zip(test_audio,train_sr)):
        inputs = feature_extractor(wv.squeeze(), return_tensors="pt",sampling_rate=48_000)
        audio_features = model.get_audio_features(**inputs)
        test_audio_feat.append(audio_features)

train_audio_feat=torch.stack(train_audio_feat).squeeze()
test_audio_feat=torch.stack(test_audio_feat).squeeze()

from sklearn.cluster import KMeans
# Choose the number of clusters
n_clusters = 10  # Example, adjust based on your data
# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(train_audio_feat)

from sklearn.manifold import TSNE
# Initialize t-SNE with 2 components (2D visualization)
tsne = TSNE(n_components=2, random_state=42)
# Apply t-SNE to the scaled features
tsne_features = tsne.fit_transform(train_audio_feat)

train_encode_model=True
thr=0.1    # thr=0.06
if train_encode_model:
    from sklearn.linear_model import RidgeCV
    voxel_models=[]
    voxels_scores=[]
    X=train_audio_feat.numpy()
    y=train_fmri_avg.numpy()

    pbar=tqdm.trange(train_fmri_avg.shape[-1],position=0)
    for v in pbar:
        vm=RidgeCV()
        y_v=y[:,v]
        vm.fit(X,y_v)
        score=vm.score(X, y_v)
        voxel_models.append(vm)
        voxels_scores.append(score)
        pbar.set_description(f"score {score}")
        
    voxels_scores=np.array(voxels_scores)
    plt.hist(voxels_scores)
    R2_img=masker.inverse_transform(np.array(voxels_scores))
    
    R2_img_smooth=nilearn.image.smooth_img(R2_img,1)
    R2_img_smooth=nilearn.image.threshold_img(R2_img_smooth,threshold=thr,cluster_threshold=100)
    R2_data=R2_img_smooth.get_fdata()

    R2_data_masked=masker.transform(R2_img_smooth)
    selected_indices=(R2_data_masked>0)
    print(selected_indices.sum())
    plotting.plot_stat_map(R2_img_smooth,display_mode="mosaic",colorbar=True,threshold=thr)
    
    #save mask
    binary_mask=(R2_data_masked>0)*1.
    mask=masker.inverse_transform(binary_mask.squeeze())
    nib.save(mask,"mask_"+subj+".nii.gz")
else:
    selected_indices=masker.transform(nib.load("mask_"+subj+".nii.gz"))





