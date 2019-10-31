from datahelper import DataSet
from arguments import argparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
from torchsummary import summary
from sklearn.metrics import mean_squared_error
import math
import time
import os
import numpy as np

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity

def forfor(a): 
    return [item for sublist in a for item in sublist] 

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
Y = np.asarray(Y)
label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
#XD, XT, Y = prepare_interaction_pairs(XD,XT,Y,label_row_inds,label_col_inds)

'''
XD = np.repeat(XD,442,axis=0)
XT = np.tile(XT,(68,1))
'''
Y = -(np.log10(np.array(Y)/(math.pow(10,9))))

test_fold, train_fold = dataset.read_sets(FLAGS)

#train set
train_rows = label_row_inds[forfor(train_fold[0:4])]
train_cols = label_col_inds[forfor(train_fold[0:4])]
train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, train_rows, train_cols)
train_drugs = np.array(train_drugs)
train_prots = np.array(train_prots)
train_X = torch.from_numpy(np.hstack((train_drugs,train_prots)))
train_Y = torch.from_numpy(np.array(train_Y))
#test set
test_rows = label_row_inds[test_fold]
test_cols = label_col_inds[test_fold]
test_drugs, test_prots,  test_Y = prepare_interaction_pairs(XD, XT, Y, test_rows, test_cols)
test_drugs = np.array(test_drugs)
test_prots = np.array(test_prots)
test_X = torch.from_numpy(np.hstack((test_drugs,test_prots)))
test_Y = torch.from_numpy(np.array(test_Y))

#print(train_X.shape, test_X.shape)
'''
X = torch.from_numpy(np.hstack((XD,XT)))

train_X = X[0:21000]
train_Y = Y[0:21000]
test_X = X[25045:30056]
test_Y = Y[25045:30056]
'''

data_train = data.TensorDataset(train_X, train_Y)
dataset_train = data.DataLoader(dataset=data_train, batch_size=256, shuffle=True, num_workers=2) 
'''
data1 = data.TensorDataset(XD,Y)
data2 = data.TensorDataset(XT,Y)
datasetD = data.DataLoader(dataset=data1, batch_size=256, shuffle=True, num_workers=0)
datasetT = data.DataLoader(dataset=data2, batch_size=256, shuffle=False, num_workers=0)
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair is not 0:
        #print(summ)
        #print(pair)
        return summ/pair
    else:
        return 0


def global_max_pooling(input):
    """Global max pooling"""
    ret, _ = torch.max(input,2)
    return ret

#embeds = nn.Embedding(256, 128)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.smi_Embeds = nn.Embedding(65,128)
        self.prot_Embeds = nn.Embedding(26,128)
        self.conv1_smi = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4)
        self.relu = F.relu
        self.conv2_smi = nn.Conv1d(in_channels=32, out_channels=32*2, kernel_size=4)
        self.conv3_smi = nn.Conv1d(in_channels=32*2, out_channels=32*3, kernel_size = 4)
        
        self.conv1_prot = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=8)
        self.conv2_prot = nn.Conv1d(in_channels=32, out_channels=32*2, kernel_size=8)
        self.conv3_prot = nn.Conv1d(in_channels=32*2, out_channels=32*3, kernel_size = 8)
        #self.maxpool = global_max_pooling()
       
        self.fc1 = nn.Linear(192,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,1)
        #self.dropout = F.dropout(p=0.1)
    
    def forward(self,XD,XT):
        #print(XT.shape)
        encode_smiles = self.smi_Embeds(Variable(torch.LongTensor(XD.type(torch.LongTensor)).to(device)))
        #print(encode_smiles.shape)
        encode_smiles = encode_smiles.permute(0,2,1)
        encode_smiles = self.conv1_smi(encode_smiles)
        encode_smiles = self.relu(encode_smiles)
        encode_smiles = self.conv2_smi(encode_smiles)
        encode_smiles = self.relu(encode_smiles)
        encode_smiles = self.conv3_smi(encode_smiles)
        encode_smiles = self.relu(encode_smiles)
        #encode_smiles = encode_smiles.permute(0,2,1)
        encode_smiles = global_max_pooling(encode_smiles)

        encode_protein = self.prot_Embeds(Variable(torch.LongTensor(XT.type(torch.LongTensor)).to(device)))
        encode_protein = encode_protein.permute(0,2,1)
        encode_protein = self.conv1_prot(encode_protein)
        encode_protein = self.relu(encode_protein)
        encode_protein = self.conv2_prot(encode_protein)
        encode_protein = self.relu(encode_protein)
        encode_protein = self.conv3_prot(encode_protein)
        encode_protein = self.relu(encode_protein)
        #encode_protein = encode_protein.permute(0,2,1)
        encode_protein = global_max_pooling(encode_protein)

        #print(encode_smiles.shape)
        #print(encode_protein.shape)

        combined = torch.cat((encode_smiles,encode_protein),-1)
        #combined = combined.reshape(256,combined.shape[1]*combined.shape[2])
        #print(combined.shape)
        out = self.relu(self.fc1(combined))
        #print(out.shape)
        out = F.dropout(out,p=0.1)
        out = self.relu(self.fc2(out))
        out = F.dropout(out,p=0.1)
        out = self.relu(self.fc3(out))
        #out = F.dropout(out,p=0.1)
        
        out = self.fc4(out)

        return out


net = Net().to(device)
#summary(net,input_size=[(256,1285)])

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_list = []
ci_list =[]
outputs_list = []

for epoch in range(100):
    avg_loss = 0
    avg_ci = 0
  
    for i,data in enumerate(dataset_train):

        inputX, y = data
        inputD = inputX[:,0:85].to(device)
        inputT = inputX[:,85:1285].to(device)

        optimizer.zero_grad()

        outputs = net(inputD,inputT)
        #print(outputs.flatten())
        loss = criterion(outputs.flatten(), y.type(torch.FloatTensor).to(device))
        loss.backward()
        optimizer.step()

        losses = loss.item()
        avg_loss += losses
        cindex = get_cindex(list(y),outputs.cpu().detach().numpy().reshape(-1))
        avg_ci += cindex

    print("epoch:", epoch, "train_loss: ",avg_loss/(i+1), "train_CI:", avg_ci/(i+1))
    
    PATH = './net.pth'
    torch.save(net.state_dict(), PATH)
    net_test = Net().to(device)
    net_test.load_state_dict(torch.load(PATH))
    for i in range(test_X.shape[0]):
        inputD = test_X[i][0:85]
        inputD = inputD.reshape([1,85]).to(device)
        inputT = test_X[i][85:1285]
        inputT = inputT.reshape([1,1200]).to(device)
        Y = test_Y[i]

        outputs = net_test(inputD,inputT)
        outputs_list.append(outputs.flatten().cpu().detach().numpy()[0])
        #print(outputs_list)

    #loss = criterion(torch.from_numpy(np.array(outputs_list)).type(torch.FloatTensor), test_Y.type(torch.FloatTensor).to(device))
    cindex = get_cindex(list(test_Y),outputs_list)
    mse = mean_squared_error(list(test_Y),outputs_list)
    print("test_MSE:", mse,"test_CI:", cindex)
    outputs_list = []


