#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:54:17 2020

@author: user
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
import pandas as pd
import argparse
os.environ['KERAS_BACKEND'] = 'tensorflow'

from tensorflow.keras import backend as K
K.set_image_data_format='channels_last'
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
from torch.autograd import Variable
import pandas as pd
from Res import resnet50

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# from skimage import circle

#Image Show
def spatial_featuremap(t_features, img, pd_coord_tissue, imscale, radius = 10, posonly=True):
    tsimg = np.zeros(img.shape[:2])
    tsimg_row = np.array(round(pd_coord_tissue.loc[:,'imgrow']*imscale), dtype=int)
    tsimg_col = np.array(round(pd_coord_tissue.loc[:,'imgcol']*imscale), dtype=int)
    for rr, cc,t in zip(tsimg_row, tsimg_col,t_features):
        r, c = draw.circle_perimeter(rr, cc, radius=10)
        if posonly:
            if t>0:
                tsimg[r,c]= t
        else:
            tsimg[r,c]=t
    return tsimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--position', type=str,default='./CID4290/spatial/tissue_positions_list.csv')
    parser.add_argument('--image', type=str,default='./CID4290/spatial/tissue_hires_image.png')
    parser.add_argument('--scale', type=float,default=0.20187746)
    parser.add_argument('--meta', type=str,default='./br_metadata_CID4290.csv')
    parser.add_argument('--outdir', type=str, default='./SPADE_output_CID4290_res/')
    parser.add_argument('--numpcs', type=int, default= 3)
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    #Param
    sz_patch = args.patchsize
    
    br_coord = pd.read_csv(args.position,
                           header=None, names= ['barcodes','tissue','row','col','imgrow','imgcol'])
    br_meta = pd.read_csv(args.meta)
    if 'seurat_clusters' not in br_meta.columns:
        print("Warning: meta data including seruat_clusters show t-SNE map of image features with clustering info")
    else:
        print('Meta data is loaded')
    br_meta_coord = pd.merge(br_meta, br_coord, how = 'inner', right_on ='barcodes' , left_on='Unnamed: 0')

   
    brimg = plt.imread(args.image)
    print('Input image dimension:', brimg.shape)  # (2000, 2000, 3)
    
    brscale = args.scale
    br_coord_tissue = br_meta_coord.loc[br_meta_coord.tissue==1,:]




    model_path = '/Users/taoyulan/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    net = resnet50()
    net.eval()
    model_dict = net.state_dict()  # 网络层的参数
    # 需要加载的预训练参数
    pretrained_dict = torch.load(model_path)
    # 删除pretrained_dict.items()中model所没有的东西
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中
    model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
    net.load_state_dict(model_dict, strict=False)  # model加载dict中的数据，更新网络的初始值
    print('成功加载预训练权重')


    
    #Image Patch   取每个位置处的小图
    tsimg_row = np.array(round(br_coord_tissue.loc[:,'imgrow']*brscale), dtype=int)
    tsimg_col = np.array(round(br_coord_tissue.loc[:,'imgcol']*brscale), dtype=int)

    tspatches = []
    ts=[]
    sz = int(sz_patch/2)
    for rr, cc in zip(tsimg_row, tsimg_col):
        tspatches.append(brimg[rr-sz:rr+sz, cc-sz:cc+sz])
    for x in tspatches:
        x = transforms.ToTensor()(x)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        with torch.no_grad():
           y = net(x).cpu()
           y = torch.squeeze(y)
           y = y.data.numpy()
           ts.append(y)
    ts_features = pd.DataFrame(ts, index=br_coord_tissue.barcodes)  # CID
    ts_features.to_csv(args.outdir + '/features.csv')


    tspatches = np.asarray(tspatches)  # (3798, 32, 32, 3)          # 32                                    3798
    print('Image to Patches done', '....patchsize is ', sz_patch, ' .... number of patches ' , tspatches.shape[0])
    if 'seurat_clusters' in br_meta.columns:
        Y = np.asarray(br_meta['seurat_clusters'])
    
    #feature extraction
    # ts_features = pretrained_model.predict(X_in)    # 神经网络降维后的512维数据
    # ts_features = pd.DataFrame(ts_features, index=br_meta_coord.barcodes)  # 3798行，512列
    # ts_features.to_csv(args.outdir + '/ts_features.csv')

    print('Image features extracted.')

    ts_tsne = TSNE(n_components=2, init='pca',perplexity=30,random_state=10).fit_transform(ts_features)
    print('t-SNE for image features ... done')
    
    #PCA
    numpcs = args.numpcs
    pca = PCA(n_components=50)
    pca.fit(ts_features)
    ts_pca = pca.transform(ts_features)
    pd_ts_pca = pd.DataFrame(ts_pca, index=br_coord_tissue.barcodes)
    pd_ts_pca.to_csv(args.outdir+'/CID4290_pc50.csv')
    print('PCs of image features are extracted')

    for ii in range(numpcs):
        tsimg = spatial_featuremap(ts_pca[:,ii], brimg, br_coord_tissue, brscale, posonly=False)
        plt.figure(figsize=(10,10))
        plt.imshow(brimg)
        plt.imshow(tsimg, alpha=0.7, cmap='bwr', vmin = -1.0, vmax=1.0)
        plt.savefig(args.outdir+'/SPADE_pc'+str(ii+1)+'.png', dpi=300)

    pd_ts_pca = pd.DataFrame(ts_pca, index = br_coord_tissue.barcodes)   # CID
    # pd_ts_pca.to_csv(args.outdir+'/ts_features_pc_CID4290_res.csv')   # 降维后的数据
    print('PCs of image features are saved')


