from datahelper import DataSet
import datahelper
from datahelper import label_sequence
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
import json
import heapq
from collections import OrderedDict

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }
ligands = json.load(open("/home/xuyt/dc/DeepDTA-master/data/davis/ligands_can.txt"), object_pairs_hook=OrderedDict)
keys_list = list(ligands.keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        out = self.fc4(out)

        return out

FLAGS = argparser()
FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

dataset = DataSet(fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                  setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE
                  seqlen = FLAGS.max_seq_len,
                  smilen = FLAGS.max_smi_len,
                  need_shuffle = False )

XD, _, _ = dataset.parse_data(FLAGS)
XD = np.asarray(XD)

protein = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL"#input("ENTER SEQUENCE:\n")
protein_embed = label_sequence(protein, 1200, CHARPROTSET)
protein_embed = np.asarray(protein_embed).reshape(-1)

PATH = './net.pth'
net = Net().to(device)
net.load_state_dict(torch.load(PATH))

output_list = []

for i in range(XD.shape[0]):
    combined_X = np.hstack((XD[i],protein_embed))
    combined_X = torch.from_numpy(combined_X)
    inputXD = combined_X[0:85].reshape([1,85]).to(device)
    inputXT = combined_X[85:1285].reshape([1,1200]).to(device)

    output = net(inputXD, inputXT)
    output_list.append(output.flatten().cpu().detach().numpy()[0])

#output_list = sorted(output_list,reverse=True)
#max_index = np.argmax(np.array(output_list))
max_index_list = list(map(output_list.index, heapq.nlargest(10, output_list)))
for i in range(len(max_index_list)):
    max_index = max_index_list[i]
    print(i+1, "match_smile:", ligands[keys_list[max_index]],"| affinity_pre:", output_list[max_index])

