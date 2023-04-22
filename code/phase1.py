import numpy as np
import scipy.spatial.distance as sd
from neighborhood import neighbor_graph, laplacian
from correspondence import Correspondence
from stiefel import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from datareader import *
import pandas as pd
import os.path
import pdb
#cuda = torch.device('cuda')
import scipy as sp
from collections import Counter
import seaborn as sns
from random import sample
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h1_sigmoid = self.linear1(x).sigmoid()
        h2_sigmoid = self.linear2(h1_sigmoid).sigmoid()
        y_pred = self.linear3(h2_sigmoid)
        return y_pred


def train_and_project(x1_np, x2_np):
    torch.manual_seed(0)

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = x1_np.shape[0], x1_np.shape[1], 512, 64, 5  # 输出的latent为3列

    model = Net(D_in, H1, H2, D_out)

    x1 = torch.from_numpy(x1_np.astype(np.float32))
    x2 = torch.from_numpy(x2_np.astype(np.float32))
    print(x1.dtype)

    adj1 = neighbor_graph(x1_np, k=5)  # neighborhood.py中,构建邻接矩阵
    adj2 = neighbor_graph(x2_np, k=5)

    # corr = Correspondence(matrix=np.eye(N))

    w1 = np.corrcoef(x1, x2)[0:x1.shape[0], x1.shape[0]:(x1.shape[0] + x2.shape[0])]
    w1[abs(w1) > 0.5] = 1
    w1[w1 != 1] = 0
    w = np.block([[w1, adj1],  # np.block创建分块矩阵   联合相似度矩阵
                  [adj2, w1.T]])

    L_np = laplacian(w, normed=False)  # neighborhood.py中
    L = torch.from_numpy(L_np.astype(np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 优化器

    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y1_pred = model(x1)   # ft
        y2_pred = model(x2)   # gt

        outputs = torch.cat((y1_pred, y2_pred), 0)  # Ft   在给定维度上对输⼊的张量序列进⾏连接操作 ,0维上下拼接

        # Project the output onto Stiefel Manifold
        u, s, v = torch.svd(outputs, some=True)  # 奇异值分解，u左v右
        proj_outputs = u @ v.t()  # F^t  @是矩阵乘法,得到论文中F-：两种数据在公共潜在流形空间统一表示

        # Compute and print loss
        print(L.dtype)       # torch.trace对角线元素之和
        loss = torch.trace(proj_outputs.t() @ L @ proj_outputs)  # 损失  论文中loss=tr(F^tLF^)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        proj_outputs.retain_grad()  # 显式地保存非叶节点的梯度

        optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
        loss.backward(retain_graph=True)

        # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
        rgrad = proj_stiefel(proj_outputs, proj_outputs.grad)  # 欧几里得梯度    stiefel.py中

        optimizer.zero_grad()
        # Backpropogate the Rimannian gradient w.r.t proj_outputs
        proj_outputs.backward(rgrad)   # 黎曼梯度

        optimizer.step()

    proj_outputs_np = proj_outputs.detach().numpy()
    return proj_outputs_np

geneExp = pd.read_csv('../CID4290data/CID4290_gene_1000.csv',index_col=0)
Efeature = pd.read_csv('../CID4290data/CID4290_pc50.csv',index_col=0)

x1_np = preprocessing.scale(geneExp.T.to_numpy())
x2_np = preprocessing.scale(Efeature.T.to_numpy())
projections = train_and_project(x1_np, x2_np)
# print(projections.shape)
projections = pd.DataFrame(projections)
features = geneExp.T.index.tolist()[0:]+Efeature.columns.tolist()
projections.index = features
projections.to_csv("../CID4290data/deep_latent_1000.csv")