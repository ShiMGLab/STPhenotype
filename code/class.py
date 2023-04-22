import numpy as np
import scipy.spatial.distance as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os.path
import pdb
#cuda = torch.device('cuda')
import scipy as sp
from torch.autograd import Variable
import torch.optim as optim
from random import sample #sample()是Python中随机模块的内置函数，可返回从序列中选择的项目的特定长度列表
import random
from imblearn.over_sampling import SMOTE  #使用imblearn进行过采样，用SMOTE方法
import seaborn as sns #Seaborn 是一个数据可视化库，可帮助在Python中创建有趣的数据可视化
from sklearn import preprocessing  # 预处理数据
from captum.attr import IntegratedGradients #模型解释库。该库为许多新的算法提供了解释性
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from mode import train_epoch,train_epoch_noreg,Net
def train_epoch(model, X_train, y_train, opt, criterion, sim, batch_size=200):
    model.train()
    sim = sim
    losses = []
    for beg_i in range(0, X_train.size(0), batch_size):
        x_batch = X_train[beg_i:beg_i + batch_size, :]
        y_batch = y_train[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch.float())
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        #print(loss)
        reg = torch.tensor(0., requires_grad=True)
        for name, param in net.fc1.named_parameters():
            if 'weight' in name:
                # M = .5 * ((torch.eye(feature_dim) - sim).T @ (torch.eye(feature_dim) - sim)) + .5 * torch.eye(feature_dim)
                M = .9 * ((torch.eye(feature_dim) - sim).T @ (torch.eye(feature_dim) - sim)) + .1 * torch.eye(feature_dim)
                # 对输入的Tensor求范数
                reg = torch.norm(reg + param @ M.float() @ param.T, 2)
                loss += reg
#         for param in model.parameters():
#             loss += .1 * torch.sum(torch.abs(param))
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses

def  train_epoch_noreg(model, X_train, y_train, opt, criterion, sim, batch_size=200):
    model.train()
    sim = sim
    losses = []
    for beg_i in range(0, X_train.size(0), batch_size):
        x_batch = X_train[beg_i:beg_i + batch_size, :]
        y_batch = y_train[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch.float())
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        #print(loss)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses

class Net(nn.Module):

    def __init__(self):

        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 200)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(200, 50)
        self.prelu = nn.PReLU(1)
        # self.out = nn.Linear(50, 5)
        self.out = nn.Linear(50, 4)  # 根据类别数目修改
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

# 读入数据
# geneExp = pd.read_csv('../dataset/cm_fin.csv', index_col=0)
geneExp = pd.read_csv('../dataset/cm_fin_1000.csv', index_col=0)
# geneExp = pd.read_csv('../CID4290data/CID4290_gene_1000.csv', index_col=0)
print('x1_np.shape:', geneExp.shape)
Efeature = pd.read_csv('../dataset/ts_pc.csv',index_col=0)
# Efeature = pd.read_csv('../CID4290data/CID4290_pc50.csv',index_col=0)
label = pd.read_csv('../dataset/label.csv')
# label = pd.read_csv('../CID4290data/label.csv')
feature_new = pd.read_csv("../dataset/deep_latent_1000.csv", index_col=0)  # 流形1
# feature_new = pd.read_csv("../CID4290data/deep_latent_1000.csv", index_col=0)  # 流形1
# ma_latent = pd.read_csv("../dataset/ma_lat_1000.csv").to_numpy()   # 流形ma
cca_latent = pd.read_csv("../dataset/cca_lat_1000.csv").to_numpy()   # 流形ma
# 直接拼在一起
# data = np.concatenate((geneExp.to_numpy(), Efeature.to_numpy()), axis=1)
# data = preprocessing.scale(data)
# 基本数据处理转化
x1_np = preprocessing.scale(geneExp.T.to_numpy())   # 改了
x2_np = preprocessing.scale(Efeature.T.to_numpy())  # 数据标准化
data = np.concatenate((x1_np, x2_np), axis=0).T  # 能够一次完成多个数组的拼接

# 距离矩阵
distance_matrix = sp.spatial.distance_matrix(feature_new, feature_new)  # 论文中 D
# distance_matrix = sp.spatial.distance_matrix(ma_latent,ma_latent)  # ma
# sim_mat = 1 / (1 + distance_matrix)  # 论文中 S
# sim_mat = 1 / (1 + distance_matrix2)  # ma
sim_mat = 1/np.exp(distance_matrix)  # S的第二种形式
sim_mat[sim_mat > np.percentile(sim_mat, 50)] = 1  # 找到一组数的分位数值，这里50%二分位
sim_mat[sim_mat != 1] = 0
sim = torch.from_numpy(sim_mat)
sim = sim.fill_diagonal_(0).float()  # 可以将numpy数组的对角线填充为参数中传递的值，这里对角线设为0

labels = label.iloc[:, 1].to_numpy()  # 行列切片以“，”隔开，前面的冒号就是取行数，后面的冒号是取列数
# train test split
acc_reg = []
acc_noreg = []
acc_e = []
acc_t = []

pred_reg = []
pred_noreg = []
pred_e = []
pred_t = []

X = data             #factorize：它可以创建一些数字，来表示类别变量，对每一个类别映射一个ID
# X = pd.DataFrame(X)
# X.to_csv("../dataset/merg.csv")
# X = pd.read_csv('../dataset/mer.csv', index_col=0).to_numpy()
y = pd.factorize(labels,sort=True)[0]  # “因式分解”，把常见的字符型变量分解为数字
print('y.shape:', y.shape)

# X, y = SMOTE(random_state = 0).fit_resample(X, y)
# print("After oversampling whole: ",Counter(y))
# X = torch.tensor(X)
# y = torch.tensor(y)

for i in range(10):   # 100改成1
    # X_train, X_vali, y_train, y_vali = train_test_split(X_train_vali,y_train_vali,test_size=0.2,
    #                                                    random_state=i, stratify = y_train_vali)
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2,
                                                        random_state=i, stratify=y)
    X_train_e, X_vali_e, y_train_e, y_vali_e = train_test_split(x1_np.T, y, test_size=0.2,
                                                                random_state=i, stratify=y)
    X_train_t, X_vali_t, y_train_t, y_vali_t = train_test_split(x2_np.T, y, test_size=0.2,
                                                                random_state=i, stratify=y)
    # print("Before oversampling: ",Counter(y_train))

    # fit and apply the transform
    X_train, y_train = SMOTE(random_state=i).fit_resample(X_train, y_train)  # LIDB数据则不要这一行代码,CID也不要
    X_train_e, y_train_e = SMOTE(random_state=i).fit_resample(X_train_e, y_train_e)
    X_train_t, y_train_t = SMOTE(random_state=i).fit_resample(X_train_t, y_train_t)
    X_vali, y_vali = SMOTE(random_state=i).fit_resample(X_vali, y_vali)    # LIDB数据则不要这一行代码,CID也不要

    # summarize class distribution
    if i == 0:
        print("After oversampling train: ", Counter(y_train))
        print("Without oversampling Validation: ", Counter(y_vali))

    torch.manual_seed(i)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_train_e = torch.tensor(X_train_e)
    y_train_e = torch.tensor(y_train_e)
    X_train_t = torch.tensor(X_train_t)
    y_train_t = torch.tensor(y_train_t)

# 四种网络
    feature_dim = 1050
    # feature_dim = 11689      # no regulized network
#     feature_dim = 11639    # 网络2 只用基因
#     feature_dim = 50  # 网络3 只用图像降维数据
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    e_losses = []
    num_epochs = 50
    for e in range(num_epochs):
        e_losses += train_epoch(net, X_train, y_train, opt, criterion, sim) # 应用流形
        # e_losses += train_epoch_noreg(net, X_train, y_train, opt, criterion, sim)  # 网络1 不用流形
        # e_losses += train_epoch_noreg(net, X_train_e, y_train_e, opt, criterion, sim)  # 网络2 gene
        # e_losses += train_epoch_noreg(net, X_train_t, y_train_t, opt, criterion, sim)  # 网络3  img
    with torch.no_grad():
        x_tensor_test = torch.from_numpy(X_vali).float()#.to(device)    # 网络1
        # x_tensor_test = torch.from_numpy(X_vali_e).float()  # .to(device)    # 网络2
        # x_tensor_test = torch.from_numpy(X_vali_t).float()  # .to(device)    # 网络3
        net.eval()
        yhat = net(x_tensor_test)
    y_pred_softmax = torch.log_softmax(yhat, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    correct_pred = np.mean([float(y_pred_tags[i] == y_vali[i]) for i in range(len(y_vali))])
    print("Round",i,"Test Accuracy (unregularized):",correct_pred)
    acc_noreg.append(np.mean(correct_pred))
    pred_noreg.append([float(y_pred_tags[i] == y_vali[i]) for i in range(len(y_vali))])

print('mean:', np.mean(acc_noreg))

from sklearn.preprocessing import label_binarize   #标签二值化
from sklearn.metrics import confusion_matrix,f1_score,recall_score,classification_report


def plot_confuse(pred, real, classes=None):
    cm = confusion_matrix(real, pred)
    #Normalise Confusion Matrix by dividing each value by the sum of that row
    cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
    #Make DataFrame from Confusion Matrix and classes
    cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    #Display Confusion Matrix
    plt.figure(figsize = (4,4), dpi = 300)
    cm_plot = sns.heatmap(cm_df, vmin = 0, vmax = 1, annot = True, fmt = '.2f', cmap = 'Blues', square = True)
    plt.title('Confusion Matrix', fontsize = 12)
    #Display axes labels
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.tight_layout()
    return cm_plot

real = label_binarize(y_vali, classes=range(4))     # 根据类别数目修改
pred = y_pred_softmax.detach().numpy()
print('pred.shape',pred.shape)
real_int = np.argmax(real, axis=1)
pred_int = np.argmax(pred, axis=1)
label_encoder = LabelEncoder()  # 将n个类别编码为0~n-1之间的整数
label_encoder.fit(list(set(label.iloc[:, 1] )))   # 每个位置 p30 p70 p100 p120
class_list = label_encoder.classes_
print('real.shape, pred.shape:',real.shape, pred.shape,class_list.shape)
print('宏平均召回率:',recall_score(real_int,pred_int,average='macro'))
print('宏平均F1-score:',f1_score(real_int,pred_int,average='macro'))
print("每个类别的精确率和召回率：")
print(classification_report(real_int, pred_int, target_names=class_list))
plot_confuse(pred_int, real_int, class_list)
# plt.savefig("confuse_.png")
# plt.show()

# ROC
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
# real = label_binarize(y, classes=range(5))
print(real.shape, pred.shape)
#pred = net(X_train.float()).detach().numpy()
for i in range(4):    # 根据类别数目修改
    fpr[i], tpr[i], _ = roc_curve(real[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(real.ravel(), pred.ravel()) #计算真正率和假正率
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"]) #计算auc的值，曲线包围的面积

layer = class_list
plt.figure(figsize=(8,12))
ax=plt.subplot(211) # 创建小图. plt.subplot(211)表示将整个图像窗口分为2行1列, 当前位置为1
sns.distplot(acc_noreg, hist = False, kde = True,kde_kws = { 'linewidth': 3,'cumulative': True})
# plt.legend([' regularization'])
ax.text(-0.1, 1.05, "B", transform=ax.transAxes,
      fontsize=20, fontweight='bold', va='top', ha='right')

lw = 2
ax=plt.subplot(212)
# ax=plt.subplot(111)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = sns.color_palette("colorblind")
for i, color in zip(range(4), colors):    # 根据类别数目修改
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of '+layer[i]+' (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.title('')
plt.legend(loc="lower right")
ax.text(-0.1, 1.05, "C", transform=ax.transAxes,
      fontsize=25, fontweight='bold', va='top', ha='right')
# plt.savefig("roc_.png")
plt.show()
