from datahelper import DataSet
from arguments import argparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time
import os
import numpy as np

FLAGS = argparser()
FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
'''
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
'''
dataset = DataSet(fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                  setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE
                  seqlen = FLAGS.max_seq_len,
                  smilen = FLAGS.max_smi_len,
                  need_shuffle = False )
FLAGS.charseqset_size = dataset.charseqset_size 
FLAGS.charsmiset_size = dataset.charsmiset_size 

XD, XT, Y = dataset.parse_data(FLAGS)
XD = np.asarray(XD)
XT = np.asarray(XT)

a = np.array([[1,2,3],[2,2,2,2]])
a = torch.from_numpy(a)
print(a.shape)
#P = np.zeros([XD.shape[0]*XT.shape[0],2,(100,10))
D = np.zeros([XD.shape[0]*XT.shape[0],1000])
#P_and_D = 

print(P.shape)
'''
P_and_D[:,0][:].astype(np.float)
P_and_D[:,1][:].astype(np.float)
'''
P_and_D = torch.from_numpy(P_and_D)
#P_and_D[1][1] = np.zeros(10)
print(type(P_and_D))
#for i in range(XD.shape[0]):


Y = torch.from_numpy(np.asarray(Y))

print(XD,XT.shape,Y.shape)
print(torch.flatten(Y).shape)
'''       
class Net(nn.Module):
    def __init__(self, input, output,m, n ):
        super(Net, self).__init__()
        self.Embedding_smi = nn.Embedding(100,128)
        self.Embedding_prot = nn.Embedding(1000,128)
        self.relu = F.relu
        self.Conv1 = nn.Conv1d()
        self.Conv2 = nn.Conv1d()
        self.Conv3 = nn.Conv1d()
        self.FC1 = nn.Linear(1024, 1024)
        self.Dropout = F.dropout
        self.FC2 = nn.Linear(1024, 512)
        self.FC3 = nn.Linear(512, 1)
'''