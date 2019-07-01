# -*- coding: utf-8 -*-

import random
import numpy as np
import scipy
import pandas as pd
import pandas
import json
import h5py
from scipy import interpolate
from bigfile import BigFile
import pdb
import os

feature_dim=2048

def readData(feat, video_name):
    video_name = "v_" + video_name
    if os.path.exists('/gruntdata/disk1/daniel/VisualSearch/activitynet_vdc/ImageData/' + video_name):
        path = '/gruntdata/disk1/daniel/VisualSearch/activitynet_vdc/ImageData/' + video_name
    else:
        path = '/gruntdata/disk1/daniel/VisualSearch/activitynet_vdc/ImageData/' + video_name + '.webm'
        video_name = video_name + '.webm'
    feature_frame = len(os.listdir(path))
    video_list = []
    for i in range(1, feature_frame + 1):
        video_list.append(video_name + '_{:0>6d}'.format(i))
    _, vecs = feat.read(video_list)
    vecs = np.array(vecs)
    return vecs

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict():
    json_data= load_json("/gruntdata/disk1/yuyan/BSN-boundary-sensitive-network.pytorch/Evaluation/data/activity_net.v1-3.min.json")
    database=json_data['database']
    out_dict={}
    for i in range(len(database.keys())):
        video_name=database.keys()[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_second']=video_info['duration']
        out_dict[video_name]=video_new_info
    return out_dict

def poolData(data,videoAnno,num_prop=100,num_bin=1,num_sample_bin=3,pool_type="mean"):
    feature_frame=len(data)
    video_second=videoAnno['duration_second']
    corrected_second=video_second
    fps=float(feature_frame)/video_second
    st=1/fps

    if len(data)==1:
        video_feature=np.stack([data]*num_prop)
        video_feature=np.reshape(video_feature,[num_prop,feature_dim])
        return video_feature

    x=[st/2+ii*st for ii in range(len(data))]
    f=scipy.interpolate.interp1d(x,data,axis=0)
        
    video_feature=[]
    zero_sample=np.zeros(num_bin*feature_dim)
    tmp_anchor_xmin=[1.0/num_prop*i for i in range(num_prop)]
    tmp_anchor_xmax=[1.0/num_prop*i for i in range(1,num_prop+1)]        
    
    num_sample=num_bin*num_sample_bin
    for idx in range(num_prop):
        xmin=max(x[0]+0.0001,tmp_anchor_xmin[idx]*corrected_second)
        xmax=min(x[-1]-0.0001,tmp_anchor_xmax[idx]*corrected_second)
        if xmax<x[0]:
            video_feature.append(zero_sample)
            continue
        if xmin>x[-1]:
            video_feature.append(zero_sample)
            continue
            
        plen=(xmax-xmin)/(num_sample-1)
        x_new=[xmin+plen*ii for ii in range(num_sample)]
        y_new=f(x_new)
        y_new_pool=[]
        for b in range(num_bin):
            tmp_y_new=y_new[num_sample_bin*b:num_sample_bin*(b+1)]
            if pool_type=="mean":
                tmp_y_new=np.mean(y_new,axis=0)
            elif pool_type=="max":
                tmp_y_new=np.max(y_new,axis=0)
            y_new_pool.append(tmp_y_new)
        y_new_pool=np.stack(y_new_pool)
        y_new_pool=np.reshape(y_new_pool,[-1])
        video_feature.append(y_new_pool)
    video_feature=np.stack(video_feature)
    return video_feature

videoDict=getDatasetDict()
videoNameList=videoDict.keys()
random.shuffle(videoNameList)
col_names=[]
for i in range(feature_dim):
    col_names.append("f"+str(i))

feat = BigFile('/gruntdata/disk1/daniel/VisualSearch/activitynet_vdc/FeatureData/pyresnext-101_rbps13k,flatten0_output,os')

for videoName in videoNameList:
    videoAnno=videoDict[videoName]
    data=readData(feat, videoName)
    print(videoName)
    videoFeature_mean=poolData(data,videoAnno,num_prop=100,num_bin=1,num_sample_bin=3,pool_type="mean")
    outDf=pd.DataFrame(videoFeature_mean,columns=col_names)
    outDf.to_csv("./resnext_csv_mean_100/"+"v_"+videoName+".csv",index=False)
