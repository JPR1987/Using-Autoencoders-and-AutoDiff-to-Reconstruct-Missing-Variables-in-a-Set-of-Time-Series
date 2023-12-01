# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:17:59 2023

@author: roche
"""



import torch.nn as nn
import torch




def network_model(inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):
    
    
    class MockupModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict({
                'lin1': nn.Linear(
                    in_features=inputsize,    
                    out_features=hiddensize1,  
                ),
                'lstm1': nn.LSTM(
                    input_size=hiddensize1,    
                    hidden_size=hiddensize2,  
                ),
                'lin2': nn.Linear(
                    in_features=hiddensize2,    
                    out_features=hiddensize3,  
                ),
                'lstm2': nn.LSTM(
                    input_size=hiddensize3,
                    hidden_size=hiddensize4,
                ),
                'lin3': nn.Linear(
                    in_features=hiddensize4,    
                    out_features=hiddensize5)
            })
            self.tanh = nn.Tanh()
            
        def forward(self, a, b, c, d):
    
            # From [batches, seqs, seq len, features]
            # to [seq len, batch data, features]
    #        x = x.view(x_seq_len, -1, x_features)
            
            rin = torch.cat((a, b, c, d), 2)
    
            # Data is fed to the LSTM
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)
            # The prediction utilizing the whole sequence is the last one
            #print(out.shape)
            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)
    #        print(f'y_pred={y_pred.size()}')
            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
    
    
    
    
    
    
    
    
    model = MockupModel()
    return model






    
    

    
