# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 08:43:28 2023

@author: roche
"""




def gen_seq(load_path, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):


    
    import numpy as np
    import pandas as pd
    import torch
    import time
    
    
    
    
    
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #   x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var



    from torch.autograd import Variable
    #seq_len <-> "How many time steps are considered."
    
    
    
    df_SineSweep_10_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__SineSweep__In10_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    

    df_SineSweep_10_6_length = len(df_SineSweep_10_6)
    df_Vin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 2:3]
    df_Vout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 3:4]
    df_Iout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 4:5]
    df_Iin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 5:6]


    
    df_Vin_SineSweep_10_6_length = len(df_Vin_SineSweep_10_6)
    df_VinB_SineSweep_10_6 = df_Vin_SineSweep_10_6.iloc[0:df_Vin_SineSweep_10_6_length, :]

    df_Iin_SineSweep_10_6_length = len(df_Iin_SineSweep_10_6)
    df_IinB_SineSweep_10_6 = df_Iin_SineSweep_10_6.iloc[0:df_Iin_SineSweep_10_6_length, :]

    df_Vout_SineSweep_10_6_length = len(df_Vout_SineSweep_10_6)
    df_VoutB_SineSweep_10_6 = df_Vout_SineSweep_10_6.iloc[0:df_Vout_SineSweep_10_6_length, :]

    df_Iout_SineSweep_10_6_length = len(df_Iout_SineSweep_10_6)
    df_IoutB_SineSweep_10_6 = df_Iout_SineSweep_10_6.iloc[0:df_Iout_SineSweep_10_6_length, :]





   









   
    
    
    
    
    
    



    df_Pulse_5b_9 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In5b_Out9__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
 
    
    df_Pulse_5b_9_length = len(df_Pulse_5b_9)
    df_Vin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 2:3]
    df_Vout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 3:4]
    df_Iout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 4:5]
    df_Iin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 5:6]
    
    
    
    df_Vin_Pulse_5b_9_length = len(df_Vin_Pulse_5b_9)
    df_VinB_Pulse_5b_9 = df_Vin_Pulse_5b_9.iloc[0:df_Vin_Pulse_5b_9_length, :]
    
    df_Iin_Pulse_5b_9_length = len(df_Iin_Pulse_5b_9)   
    df_IinB_Pulse_5b_9 = df_Iin_Pulse_5b_9.iloc[0:df_Iin_Pulse_5b_9_length, :]  
    
    df_Vout_Pulse_5b_9_length = len(df_Vout_Pulse_5b_9)
    df_VoutB_Pulse_5b_9 = df_Vout_Pulse_5b_9.iloc[0:df_Vout_Pulse_5b_9_length, :]
        
    df_Iout_Pulse_5b_9_length = len(df_Iout_Pulse_5b_9)
    df_IoutB_Pulse_5b_9 = df_Iout_Pulse_5b_9.iloc[0:df_Iout_Pulse_5b_9_length, :]
    
    
    
    
    
  
    
    
    
    
    
    
    
    
        
        
    
    
    
    


    df_Pulse_6c_5 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In6c_Out5__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_Pulse_6c_5_length = len(df_Pulse_6c_5)
    df_Vin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 2:3]
    df_Vout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 3:4]
    df_Iout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 4:5]
    df_Iin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 5:6]
    
    
    
    df_Vin_Pulse_6c_5_length = len(df_Vin_Pulse_6c_5)
    df_VinB_Pulse_6c_5 = df_Vin_Pulse_6c_5.iloc[0:df_Vin_Pulse_6c_5_length, :]

    df_Iin_Pulse_6c_5_length = len(df_Iin_Pulse_6c_5)
    df_IinB_Pulse_6c_5 = df_Iin_Pulse_6c_5.iloc[0:df_Iin_Pulse_6c_5_length, :]
    
    df_Vout_Pulse_6c_5_length = len(df_Vout_Pulse_6c_5)
    df_VoutB_Pulse_6c_5 = df_Vout_Pulse_6c_5.iloc[0:df_Vout_Pulse_6c_5_length, :]
    
    df_Iout_Pulse_6c_5_length = len(df_Iout_Pulse_6c_5)
    df_IoutB_Pulse_6c_5 = df_Iout_Pulse_6c_5.iloc[0:df_Iout_Pulse_6c_5_length, :]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    from sklearn.preprocessing import MinMaxScaler

    scaler_Vin = MinMaxScaler()
    all_Vin_Pulse_6c_5 = scaler_Vin.fit_transform(df_VinB_Pulse_6c_5)


    scaler_Iout = MinMaxScaler()
    all_Iout_Pulse_5b_9 = scaler_Iout.fit_transform(df_IoutB_Pulse_5b_9)







    scaler_Vout = MinMaxScaler()
    all_Vout_Pulse_6c_5 = scaler_Vout.fit_transform(df_VoutB_Pulse_6c_5)


    scaler_Iin = MinMaxScaler()
    all_Iin_Pulse_6c_5 = scaler_Iin.fit_transform(df_IinB_Pulse_6c_5)
    
    
    
    
    
    
    all_Vin_SineSweep_10_6 = scaler_Vin.transform(df_VinB_SineSweep_10_6)
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    import torch.nn as nn
    import torch.optim as optim


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
          
            rin = torch.cat((a, b, c, d), 2)

            # Data is fed to the NN
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)


            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)
    
            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
        
        
        
        
        
        
        
    def generate_sequence(scaler_a, scaler_b, scaler_c, scaler_d, model, xa_sample, xb_sample, xc_sample, xd_sample):
       
        y_pred_tensor_a, y_pred_tensor_b, y_pred_tensor_c, y_pred_tensor_d = model(xa_sample, xb_sample, xc_sample, xd_sample)
        y_pred_red_a = y_pred_tensor_a[:, -1]
        y_pred_a = y_pred_red_a.cpu().tolist()
        y_pred_a = scaler_a.inverse_transform(y_pred_a)
        y_pred_red_b = y_pred_tensor_b[:, -1]
        y_pred_b = y_pred_red_b.cpu().tolist()
        y_pred_b = scaler_b.inverse_transform(y_pred_b) 
        y_pred_red_c = y_pred_tensor_c[:, -1]
        y_pred_c = y_pred_red_c.cpu().tolist()
        y_pred_c = scaler_c.inverse_transform(y_pred_c)
        y_pred_red_d = y_pred_tensor_d[:, -1]
        y_pred_d = y_pred_red_d.cpu().tolist()
        y_pred_d = scaler_d.inverse_transform(y_pred_d)
            
        return y_pred_a, y_pred_b, y_pred_c, y_pred_d
        
        
        
        
        
        
        
        
    model = MockupModel()

    #lossfn = nn.MSELoss(reduction='sum')

    learning_rate = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    
    
    
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    
    start_time = time.time()
    ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6 = generate_sequence(scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, model, Vin_all_SineSweep_10_6, Iin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iout_all_SineSweep_10_6)
    predTime = time.time() - start_time
    print("Calculation Time (generate sequence with cpu):", predTime)
    
    
    
    return ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6
    
    
  















def gen_seq_recon_Vin(save_path_reconstruction_Vin_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):
    
    import numpy as np
    import pandas as pd
    import torch
    import time
    
    device = torch.device("cpu")
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var



    from torch.autograd import Variable
    #seq_len <-> "How many time steps are considered."
    
    
    
    df_SineSweep_10_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__SineSweep__In10_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    

    df_SineSweep_10_6_length = len(df_SineSweep_10_6)
    df_Vin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 2:3]
    df_Vout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 3:4]
    df_Iout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 4:5]
    df_Iin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 5:6]


    
    df_Vin_SineSweep_10_6_length = len(df_Vin_SineSweep_10_6)
    df_VinB_SineSweep_10_6 = df_Vin_SineSweep_10_6.iloc[0:df_Vin_SineSweep_10_6_length, :]

    df_Iin_SineSweep_10_6_length = len(df_Iin_SineSweep_10_6)
    df_IinB_SineSweep_10_6 = df_Iin_SineSweep_10_6.iloc[0:df_Iin_SineSweep_10_6_length, :]

    df_Vout_SineSweep_10_6_length = len(df_Vout_SineSweep_10_6)
    df_VoutB_SineSweep_10_6 = df_Vout_SineSweep_10_6.iloc[0:df_Vout_SineSweep_10_6_length, :]

    df_Iout_SineSweep_10_6_length = len(df_Iout_SineSweep_10_6)
    df_IoutB_SineSweep_10_6 = df_Iout_SineSweep_10_6.iloc[0:df_Iout_SineSweep_10_6_length, :]





   









   
    
    
    
    
    
    
    



    df_Pulse_5b_9 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In5b_Out9__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')

    
    df_Pulse_5b_9_length = len(df_Pulse_5b_9)
    df_Vin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 2:3]
    df_Vout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 3:4]
    df_Iout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 4:5]
    df_Iin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 5:6]
    
    

    df_Vin_Pulse_5b_9_length = len(df_Vin_Pulse_5b_9)
    df_VinB_Pulse_5b_9 = df_Vin_Pulse_5b_9.iloc[0:df_Vin_Pulse_5b_9_length, :]
    
    df_Iin_Pulse_5b_9_length = len(df_Iin_Pulse_5b_9)
    df_IinB_Pulse_5b_9 = df_Iin_Pulse_5b_9.iloc[0:df_Iin_Pulse_5b_9_length, :]
    
    df_Vout_Pulse_5b_9_length = len(df_Vout_Pulse_5b_9)
    df_VoutB_Pulse_5b_9 = df_Vout_Pulse_5b_9.iloc[0:df_Vout_Pulse_5b_9_length, :]
    
    df_Iout_Pulse_5b_9_length = len(df_Iout_Pulse_5b_9)
    df_IoutB_Pulse_5b_9 = df_Iout_Pulse_5b_9.iloc[0:df_Iout_Pulse_5b_9_length, :]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
        
        
    
    
    
    


    df_Pulse_6c_5 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In6c_Out5__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')

    
    df_Pulse_6c_5_length = len(df_Pulse_6c_5)
    df_Vin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 2:3]
    df_Vout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 3:4]
    df_Iout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 4:5]
    df_Iin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 5:6]
    
    

    df_Vin_Pulse_6c_5_length = len(df_Vin_Pulse_6c_5)
    df_VinB_Pulse_6c_5 = df_Vin_Pulse_6c_5.iloc[0:df_Vin_Pulse_6c_5_length, :]
    
    df_Iin_Pulse_6c_5_length = len(df_Iin_Pulse_6c_5)
    df_IinB_Pulse_6c_5 = df_Iin_Pulse_6c_5.iloc[0:df_Iin_Pulse_6c_5_length, :]
    
    df_Vout_Pulse_6c_5_length = len(df_Vout_Pulse_6c_5)
    df_VoutB_Pulse_6c_5 = df_Vout_Pulse_6c_5.iloc[0:df_Vout_Pulse_6c_5_length, :]
    
    df_Iout_Pulse_6c_5_length = len(df_Iout_Pulse_6c_5)
    df_IoutB_Pulse_6c_5 = df_Iout_Pulse_6c_5.iloc[0:df_Iout_Pulse_6c_5_length, :]
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    from sklearn.preprocessing import MinMaxScaler


    scaler_Vin = MinMaxScaler()
    all_Vin_Pulse_6c_5 = scaler_Vin.fit_transform(df_VinB_Pulse_6c_5)


    scaler_Iout = MinMaxScaler()
    all_Iout_Pulse_5b_9 = scaler_Iout.fit_transform(df_IoutB_Pulse_5b_9)







    scaler_Vout = MinMaxScaler()
    all_Vout_Pulse_6c_5 = scaler_Vout.fit_transform(df_VoutB_Pulse_6c_5)


    scaler_Iin = MinMaxScaler()
    all_Iin_Pulse_6c_5 = scaler_Iin.fit_transform(df_IinB_Pulse_6c_5)
    
    
    
    
    
    
    all_Vin_SineSweep_10_6 = scaler_Vin.transform(df_VinB_SineSweep_10_6)
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    import torch.nn as nn
    import torch.optim as optim


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
           
            rin = torch.cat((a, b, c, d), 2)

            # Data is fed to the NN
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)

            #print(out.shape)
            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)
 
            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
        
        
        
        
        
        
        
    def generate_sequence(scaler_a, scaler_b, scaler_c, scaler_d, model, xa_sample, xb_sample, xc_sample, xd_sample):
        
        y_pred_tensor_a, y_pred_tensor_b, y_pred_tensor_c, y_pred_tensor_d = model(xa_sample, xb_sample, xc_sample, xd_sample)
        y_pred_red_a = y_pred_tensor_a[:, -1]
        y_pred_a = y_pred_red_a.cpu().tolist()
        y_pred_a = scaler_a.inverse_transform(y_pred_a)
        y_pred_red_b = y_pred_tensor_b[:, -1]
        y_pred_b = y_pred_red_b.cpu().tolist()
        y_pred_b = scaler_b.inverse_transform(y_pred_b) 
        y_pred_red_c = y_pred_tensor_c[:, -1]
        y_pred_c = y_pred_red_c.cpu().tolist()
        y_pred_c = scaler_c.inverse_transform(y_pred_c)
        y_pred_red_d = y_pred_tensor_d[:, -1]
        y_pred_d = y_pred_red_d.cpu().tolist()
        y_pred_d = scaler_d.inverse_transform(y_pred_d)
            
        return y_pred_a, y_pred_b, y_pred_c, y_pred_d
        
        
        
        
        
        
        
        
    model = MockupModel()

    #lossfn = nn.MSELoss(reduction='sum')

    learning_rate = 5e-3

   
    
    
    ADVar1_init = Variable(torch.zeros(500000, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True

    optimizer = optim.Adam([ADVar1_seq], lr=learning_rate)




    
    checkpoint = torch.load(save_path_reconstruction_Vin_left, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #run = checkpoint['run']
    ADVar1_seq = checkpoint['ADVar1_seq']
    
    
    
    
    
    A_app_all = ADVar1_seq.to(device)
    B_app_all = Iin_all_SineSweep_10_6.to(device)
    C_app_all = Vout_all_SineSweep_10_6.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)





    start_time = time.time()
    ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6 = generate_sequence(scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, model, A_app_all, B_app_all, C_app_all, D_app_all)
    predTime = time.time() - start_time
    print("Calculation Time (generate sequence with cpu):", predTime)
    
    
    
    return ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6
    


















def gen_seq_recon_Iin(save_path_reconstruction_Iin_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):
   
    import numpy as np
    import pandas as pd
    import torch
    import time
    
    
    device = torch.device("cpu") 
    
    
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var



    from torch.autograd import Variable
    #seq_len <->"How many time steps are considered."
    
    
    
    df_SineSweep_10_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__SineSweep__In10_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    

    df_SineSweep_10_6_length = len(df_SineSweep_10_6)
    df_Vin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 2:3]
    df_Vout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 3:4]
    df_Iout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 4:5]
    df_Iin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 5:6]


    
    df_Vin_SineSweep_10_6_length = len(df_Vin_SineSweep_10_6)
    df_VinB_SineSweep_10_6 = df_Vin_SineSweep_10_6.iloc[0:df_Vin_SineSweep_10_6_length, :]

    df_Iin_SineSweep_10_6_length = len(df_Iin_SineSweep_10_6)
    df_IinB_SineSweep_10_6 = df_Iin_SineSweep_10_6.iloc[0:df_Iin_SineSweep_10_6_length, :]

    df_Vout_SineSweep_10_6_length = len(df_Vout_SineSweep_10_6)
    df_VoutB_SineSweep_10_6 = df_Vout_SineSweep_10_6.iloc[0:df_Vout_SineSweep_10_6_length, :]

    df_Iout_SineSweep_10_6_length = len(df_Iout_SineSweep_10_6)
    df_IoutB_SineSweep_10_6 = df_Iout_SineSweep_10_6.iloc[0:df_Iout_SineSweep_10_6_length, :]







    
    
    
    
    
    
    



    df_Pulse_5b_9 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In5b_Out9__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')

    
    df_Pulse_5b_9_length = len(df_Pulse_5b_9)
    df_Vin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 2:3]
    df_Vout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 3:4]
    df_Iout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 4:5]
    df_Iin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 5:6]
    
    

    df_Vin_Pulse_5b_9_length = len(df_Vin_Pulse_5b_9)
    df_VinB_Pulse_5b_9 = df_Vin_Pulse_5b_9.iloc[0:df_Vin_Pulse_5b_9_length, :]
    
    df_Iin_Pulse_5b_9_length = len(df_Iin_Pulse_5b_9)
    df_IinB_Pulse_5b_9 = df_Iin_Pulse_5b_9.iloc[0:df_Iin_Pulse_5b_9_length, :]
    
    df_Vout_Pulse_5b_9_length = len(df_Vout_Pulse_5b_9)
    df_VoutB_Pulse_5b_9 = df_Vout_Pulse_5b_9.iloc[0:df_Vout_Pulse_5b_9_length, :]
    
    df_Iout_Pulse_5b_9_length = len(df_Iout_Pulse_5b_9)
    df_IoutB_Pulse_5b_9 = df_Iout_Pulse_5b_9.iloc[0:df_Iout_Pulse_5b_9_length, :]
    
    
    
    
  
    
    
    
    
    
    
  
        
    
    
    
    


    df_Pulse_6c_5 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In6c_Out5__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')

    
    df_Pulse_6c_5_length = len(df_Pulse_6c_5)
    df_Vin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 2:3]
    df_Vout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 3:4]
    df_Iout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 4:5]
    df_Iin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 5:6]
    

    df_Vin_Pulse_6c_5_length = len(df_Vin_Pulse_6c_5)
    df_VinB_Pulse_6c_5 = df_Vin_Pulse_6c_5.iloc[0:df_Vin_Pulse_6c_5_length, :]
    
    df_Iin_Pulse_6c_5_length = len(df_Iin_Pulse_6c_5)
    df_IinB_Pulse_6c_5 = df_Iin_Pulse_6c_5.iloc[0:df_Iin_Pulse_6c_5_length, :]
    
    df_Vout_Pulse_6c_5_length = len(df_Vout_Pulse_6c_5)
    df_VoutB_Pulse_6c_5 = df_Vout_Pulse_6c_5.iloc[0:df_Vout_Pulse_6c_5_length, :]
    
    df_Iout_Pulse_6c_5_length = len(df_Iout_Pulse_6c_5)
    df_IoutB_Pulse_6c_5 = df_Iout_Pulse_6c_5.iloc[0:df_Iout_Pulse_6c_5_length, :]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    from sklearn.preprocessing import MinMaxScaler


    scaler_Vin = MinMaxScaler()
    all_Vin_Pulse_6c_5 = scaler_Vin.fit_transform(df_VinB_Pulse_6c_5)


    scaler_Iout = MinMaxScaler()
    all_Iout_Pulse_5b_9 = scaler_Iout.fit_transform(df_IoutB_Pulse_5b_9)







    scaler_Vout = MinMaxScaler()
    all_Vout_Pulse_6c_5 = scaler_Vout.fit_transform(df_VoutB_Pulse_6c_5)


    scaler_Iin = MinMaxScaler()
    all_Iin_Pulse_6c_5 = scaler_Iin.fit_transform(df_IinB_Pulse_6c_5)
    
    
    
    
    
    
    all_Vin_SineSweep_10_6 = scaler_Vin.transform(df_VinB_SineSweep_10_6)
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    import torch.nn as nn
    import torch.optim as optim


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

            
           
            rin = torch.cat((a, b, c, d), 2)

            # Data is fed to the NN
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)
    
            #print(out.shape)
            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)
   
            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
        
        
        
        
        
        
        
    def generate_sequence(scaler_a, scaler_b, scaler_c, scaler_d, model, xa_sample, xb_sample, xc_sample, xd_sample):
        
        y_pred_tensor_a, y_pred_tensor_b, y_pred_tensor_c, y_pred_tensor_d = model(xa_sample, xb_sample, xc_sample, xd_sample)
        y_pred_red_a = y_pred_tensor_a[:, -1]
        y_pred_a = y_pred_red_a.cpu().tolist()
        y_pred_a = scaler_a.inverse_transform(y_pred_a)
        y_pred_red_b = y_pred_tensor_b[:, -1]
        y_pred_b = y_pred_red_b.cpu().tolist()
        y_pred_b = scaler_b.inverse_transform(y_pred_b) 
        y_pred_red_c = y_pred_tensor_c[:, -1]
        y_pred_c = y_pred_red_c.cpu().tolist()
        y_pred_c = scaler_c.inverse_transform(y_pred_c)
        y_pred_red_d = y_pred_tensor_d[:, -1]
        y_pred_d = y_pred_red_d.cpu().tolist()
        y_pred_d = scaler_d.inverse_transform(y_pred_d)
            
        return y_pred_a, y_pred_b, y_pred_c, y_pred_d
        
        
        
        
        
        
        
        
    model = MockupModel()

    #lossfn = nn.MSELoss(reduction='sum')

    learning_rate = 5e-3

   
    
    
    ADVar1_init = Variable(torch.zeros(500000, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True

    optimizer = optim.Adam([ADVar1_seq], lr=learning_rate)




    
    checkpoint = torch.load(save_path_reconstruction_Iin_left, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #run = checkpoint['run']
    ADVar1_seq = checkpoint['ADVar1_seq']
    
    
    
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = ADVar1_seq.to(device)
    C_app_all = Vout_all_SineSweep_10_6.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)





    start_time = time.time()
    ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6 = generate_sequence(scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, model, A_app_all, B_app_all, C_app_all, D_app_all)
    predTime = time.time() - start_time
    print("Calculation Time (generate sequence with cpu):", predTime)
    
    
    
    return ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6













def gen_seq_recon_Vout(save_path_reconstruction_Vout_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):
 
    import numpy as np
    import pandas as pd
    import torch
    import time
    
    
    device = torch.device("cpu") 
    
    
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var



    from torch.autograd import Variable
    #seq_len <-> "How many time steps are considered."
    
    
    
    df_SineSweep_10_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__SineSweep__In10_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    

    df_SineSweep_10_6_length = len(df_SineSweep_10_6)
    df_Vin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 2:3]
    df_Vout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 3:4]
    df_Iout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 4:5]
    df_Iin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 5:6]


    
    df_Vin_SineSweep_10_6_length = len(df_Vin_SineSweep_10_6)
    df_VinB_SineSweep_10_6 = df_Vin_SineSweep_10_6.iloc[0:df_Vin_SineSweep_10_6_length, :]

    df_Iin_SineSweep_10_6_length = len(df_Iin_SineSweep_10_6)
    df_IinB_SineSweep_10_6 = df_Iin_SineSweep_10_6.iloc[0:df_Iin_SineSweep_10_6_length, :]

    df_Vout_SineSweep_10_6_length = len(df_Vout_SineSweep_10_6)
    df_VoutB_SineSweep_10_6 = df_Vout_SineSweep_10_6.iloc[0:df_Vout_SineSweep_10_6_length, :]

    df_Iout_SineSweep_10_6_length = len(df_Iout_SineSweep_10_6)
    df_IoutB_SineSweep_10_6 = df_Iout_SineSweep_10_6.iloc[0:df_Iout_SineSweep_10_6_length, :]





   









   
    
    
    
    
    
    



    df_Pulse_5b_9 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In5b_Out9__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
   
    
    df_Pulse_5b_9_length = len(df_Pulse_5b_9)
    df_Vin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 2:3]
    df_Vout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 3:4]
    df_Iout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 4:5]
    df_Iin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 5:6]
    
    

    df_Vin_Pulse_5b_9_length = len(df_Vin_Pulse_5b_9)
    df_VinB_Pulse_5b_9 = df_Vin_Pulse_5b_9.iloc[0:df_Vin_Pulse_5b_9_length, :]
    
    df_Iin_Pulse_5b_9_length = len(df_Iin_Pulse_5b_9)
    df_IinB_Pulse_5b_9 = df_Iin_Pulse_5b_9.iloc[0:df_Iin_Pulse_5b_9_length, :]
    
    df_Vout_Pulse_5b_9_length = len(df_Vout_Pulse_5b_9)
    df_VoutB_Pulse_5b_9 = df_Vout_Pulse_5b_9.iloc[0:df_Vout_Pulse_5b_9_length, :]
    
    df_Iout_Pulse_5b_9_length = len(df_Iout_Pulse_5b_9)
    df_IoutB_Pulse_5b_9 = df_Iout_Pulse_5b_9.iloc[0:df_Iout_Pulse_5b_9_length, :]
    
    
    
    
    
   
    
   
        
        
    
    
    
    


    df_Pulse_6c_5 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In6c_Out5__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
   
    
    df_Pulse_6c_5_length = len(df_Pulse_6c_5)
    df_Vin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 2:3]
    df_Vout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 3:4]
    df_Iout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 4:5]
    df_Iin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 5:6]
    
    
    
    df_Vin_Pulse_6c_5_length = len(df_Vin_Pulse_6c_5)
    df_VinB_Pulse_6c_5 = df_Vin_Pulse_6c_5.iloc[0:df_Vin_Pulse_6c_5_length, :]
    
    df_Iin_Pulse_6c_5_length = len(df_Iin_Pulse_6c_5)
    df_IinB_Pulse_6c_5 = df_Iin_Pulse_6c_5.iloc[0:df_Iin_Pulse_6c_5_length, :]
    
    df_Vout_Pulse_6c_5_length = len(df_Vout_Pulse_6c_5)
    df_VoutB_Pulse_6c_5 = df_Vout_Pulse_6c_5.iloc[0:df_Vout_Pulse_6c_5_length, :]
    
    df_Iout_Pulse_6c_5_length = len(df_Iout_Pulse_6c_5)
    df_IoutB_Pulse_6c_5 = df_Iout_Pulse_6c_5.iloc[0:df_Iout_Pulse_6c_5_length, :]
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    from sklearn.preprocessing import MinMaxScaler


    scaler_Vin = MinMaxScaler()
    all_Vin_Pulse_6c_5 = scaler_Vin.fit_transform(df_VinB_Pulse_6c_5)


    scaler_Iout = MinMaxScaler()
    all_Iout_Pulse_5b_9 = scaler_Iout.fit_transform(df_IoutB_Pulse_5b_9)







    scaler_Vout = MinMaxScaler()
    all_Vout_Pulse_6c_5 = scaler_Vout.fit_transform(df_VoutB_Pulse_6c_5)


    scaler_Iin = MinMaxScaler()
    all_Iin_Pulse_6c_5 = scaler_Iin.fit_transform(df_IinB_Pulse_6c_5)
    
    
    
    
    
    
    all_Vin_SineSweep_10_6 = scaler_Vin.transform(df_VinB_SineSweep_10_6)
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    import torch.nn as nn
    import torch.optim as optim


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


           
            rin = torch.cat((a, b, c, d), 2)

            # Data is fed to the NN
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)
 
            #print(out.shape)
            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)

            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
        
        
        
        
        
        
        
    def generate_sequence(scaler_a, scaler_b, scaler_c, scaler_d, model, xa_sample, xb_sample, xc_sample, xd_sample):

        y_pred_tensor_a, y_pred_tensor_b, y_pred_tensor_c, y_pred_tensor_d = model(xa_sample, xb_sample, xc_sample, xd_sample)
        y_pred_red_a = y_pred_tensor_a[:, -1]
        y_pred_a = y_pred_red_a.cpu().tolist()
        y_pred_a = scaler_a.inverse_transform(y_pred_a)
        y_pred_red_b = y_pred_tensor_b[:, -1]
        y_pred_b = y_pred_red_b.cpu().tolist()
        y_pred_b = scaler_b.inverse_transform(y_pred_b) 
        y_pred_red_c = y_pred_tensor_c[:, -1]
        y_pred_c = y_pred_red_c.cpu().tolist()
        y_pred_c = scaler_c.inverse_transform(y_pred_c)
        y_pred_red_d = y_pred_tensor_d[:, -1]
        y_pred_d = y_pred_red_d.cpu().tolist()
        y_pred_d = scaler_d.inverse_transform(y_pred_d)
            
        return y_pred_a, y_pred_b, y_pred_c, y_pred_d
        
        
        
        
        
        
        
        
    model = MockupModel()

    #lossfn = nn.MSELoss(reduction='sum')

    learning_rate = 5e-3

   
    
    
    ADVar1_init = Variable(torch.zeros(500000, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True

    optimizer = optim.Adam([ADVar1_seq], lr=learning_rate)




    
    checkpoint = torch.load(save_path_reconstruction_Vout_left, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #run = checkpoint['run']
    ADVar1_seq = checkpoint['ADVar1_seq']
    
    
    
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = Iin_all_SineSweep_10_6.to(device)
    C_app_all = ADVar1_seq.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)





    start_time = time.time()
    ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6 = generate_sequence(scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, model, A_app_all, B_app_all, C_app_all, D_app_all)
    predTime = time.time() - start_time
    print("Calculation Time (generate sequence with cpu):", predTime)
    
    
    
    return ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6









def gen_seq_recon_Iout(save_path_reconstruction_Iout_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):
 
    import numpy as np
    import pandas as pd
    import torch
    import time
    
    
    device = torch.device("cpu") 
    
    
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var



    from torch.autograd import Variable
    #seq_len <-> "How many time steps are considered."
    
    
    
    df_SineSweep_10_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__SineSweep__In10_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    

    df_SineSweep_10_6_length = len(df_SineSweep_10_6)
    df_Vin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 2:3]
    df_Vout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 3:4]
    df_Iout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 4:5]
    df_Iin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 5:6]


    
    df_Vin_SineSweep_10_6_length = len(df_Vin_SineSweep_10_6)
    df_VinB_SineSweep_10_6 = df_Vin_SineSweep_10_6.iloc[0:df_Vin_SineSweep_10_6_length, :]

    df_Iin_SineSweep_10_6_length = len(df_Iin_SineSweep_10_6)
    df_IinB_SineSweep_10_6 = df_Iin_SineSweep_10_6.iloc[0:df_Iin_SineSweep_10_6_length, :]

    df_Vout_SineSweep_10_6_length = len(df_Vout_SineSweep_10_6)
    df_VoutB_SineSweep_10_6 = df_Vout_SineSweep_10_6.iloc[0:df_Vout_SineSweep_10_6_length, :]

    df_Iout_SineSweep_10_6_length = len(df_Iout_SineSweep_10_6)
    df_IoutB_SineSweep_10_6 = df_Iout_SineSweep_10_6.iloc[0:df_Iout_SineSweep_10_6_length, :]







    
    
    
    
    



    df_Pulse_5b_9 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In5b_Out9__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_Pulse_5b_9_length = len(df_Pulse_5b_9)
    df_Vin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 2:3]
    df_Vout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 3:4]
    df_Iout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 4:5]
    df_Iin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 5:6]
    
    

    df_Vin_Pulse_5b_9_length = len(df_Vin_Pulse_5b_9)
    df_VinB_Pulse_5b_9 = df_Vin_Pulse_5b_9.iloc[0:df_Vin_Pulse_5b_9_length, :]
    
    df_Iin_Pulse_5b_9_length = len(df_Iin_Pulse_5b_9)
    df_IinB_Pulse_5b_9 = df_Iin_Pulse_5b_9.iloc[0:df_Iin_Pulse_5b_9_length, :]
    
    df_Vout_Pulse_5b_9_length = len(df_Vout_Pulse_5b_9)
    df_VoutB_Pulse_5b_9 = df_Vout_Pulse_5b_9.iloc[0:df_Vout_Pulse_5b_9_length, :]
    
    df_Iout_Pulse_5b_9_length = len(df_Iout_Pulse_5b_9)
    df_IoutB_Pulse_5b_9 = df_Iout_Pulse_5b_9.iloc[0:df_Iout_Pulse_5b_9_length, :]
    
    
    
    
    
    
    
    
    
    
    


    df_Pulse_6c_5 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In6c_Out5__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_Pulse_6c_5_length = len(df_Pulse_6c_5)
    df_Vin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 2:3]
    df_Vout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 3:4]
    df_Iout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 4:5]
    df_Iin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 5:6]
    
    

    df_Vin_Pulse_6c_5_length = len(df_Vin_Pulse_6c_5)
    df_VinB_Pulse_6c_5 = df_Vin_Pulse_6c_5.iloc[0:df_Vin_Pulse_6c_5_length, :]
    
    df_Iin_Pulse_6c_5_length = len(df_Iin_Pulse_6c_5)
    df_IinB_Pulse_6c_5 = df_Iin_Pulse_6c_5.iloc[0:df_Iin_Pulse_6c_5_length, :]
    
    df_Vout_Pulse_6c_5_length = len(df_Vout_Pulse_6c_5)
    df_VoutB_Pulse_6c_5 = df_Vout_Pulse_6c_5.iloc[0:df_Vout_Pulse_6c_5_length, :]
    
    df_Iout_Pulse_6c_5_length = len(df_Iout_Pulse_6c_5)
    df_IoutB_Pulse_6c_5 = df_Iout_Pulse_6c_5.iloc[0:df_Iout_Pulse_6c_5_length, :]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    from sklearn.preprocessing import MinMaxScaler


    scaler_Vin = MinMaxScaler()
    all_Vin_Pulse_6c_5 = scaler_Vin.fit_transform(df_VinB_Pulse_6c_5)


    scaler_Iout = MinMaxScaler()
    all_Iout_Pulse_5b_9 = scaler_Iout.fit_transform(df_IoutB_Pulse_5b_9)







    scaler_Vout = MinMaxScaler()
    all_Vout_Pulse_6c_5 = scaler_Vout.fit_transform(df_VoutB_Pulse_6c_5)


    scaler_Iin = MinMaxScaler()
    all_Iin_Pulse_6c_5 = scaler_Iin.fit_transform(df_IinB_Pulse_6c_5)
    
    
    
    
    
    
    all_Vin_SineSweep_10_6 = scaler_Vin.transform(df_VinB_SineSweep_10_6)
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    import torch.nn as nn
    import torch.optim as optim


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

            
            rin = torch.cat((a, b, c, d), 2)

            # Data is fed to the NN
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)
    
            #print(out.shape)
            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)
    
            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
        
        
        
        
        
        
        
    def generate_sequence(scaler_a, scaler_b, scaler_c, scaler_d, model, xa_sample, xb_sample, xc_sample, xd_sample):
        
        y_pred_tensor_a, y_pred_tensor_b, y_pred_tensor_c, y_pred_tensor_d = model(xa_sample, xb_sample, xc_sample, xd_sample)
        y_pred_red_a = y_pred_tensor_a[:, -1]
        y_pred_a = y_pred_red_a.cpu().tolist()
        y_pred_a = scaler_a.inverse_transform(y_pred_a)
        y_pred_red_b = y_pred_tensor_b[:, -1]
        y_pred_b = y_pred_red_b.cpu().tolist()
        y_pred_b = scaler_b.inverse_transform(y_pred_b) 
        y_pred_red_c = y_pred_tensor_c[:, -1]
        y_pred_c = y_pred_red_c.cpu().tolist()
        y_pred_c = scaler_c.inverse_transform(y_pred_c)
        y_pred_red_d = y_pred_tensor_d[:, -1]
        y_pred_d = y_pred_red_d.cpu().tolist()
        y_pred_d = scaler_d.inverse_transform(y_pred_d)
            
        return y_pred_a, y_pred_b, y_pred_c, y_pred_d
        
        
        
        
        
        
        
        
    model = MockupModel()

    #lossfn = nn.MSELoss(reduction='sum')

    learning_rate = 5e-3

   
    
    
    ADVar1_init = Variable(torch.zeros(500000, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True

    optimizer = optim.Adam([ADVar1_seq], lr=learning_rate)




    
    checkpoint = torch.load(save_path_reconstruction_Iout_left, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #run = checkpoint['run']
    ADVar1_seq = checkpoint['ADVar1_seq']
    
    
    
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = Iin_all_SineSweep_10_6.to(device)
    C_app_all = Vout_all_SineSweep_10_6.to(device)
    D_app_all = ADVar1_seq.to(device)





    start_time = time.time()
    ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6 = generate_sequence(scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, model, A_app_all, B_app_all, C_app_all, D_app_all)
    predTime = time.time() - start_time
    print("Calculation Time (generate sequence with cpu):", predTime)
    
    
    
    return ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6


















def gen_seq_recon_Iin_Vout(save_path_reconstruction_Iin_Vout_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5):
    
    import numpy as np
    import pandas as pd
    import torch
    import time
    
    
    device = torch.device("cpu") 
    
    
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var



    from torch.autograd import Variable
    #seq_len <-> "How many time steps are considered."
    
    
    
    df_SineSweep_10_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__SineSweep__In10_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    

    df_SineSweep_10_6_length = len(df_SineSweep_10_6)
    df_Vin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 2:3]
    df_Vout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 3:4]
    df_Iout_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 4:5]
    df_Iin_SineSweep_10_6 = df_SineSweep_10_6.iloc[0:df_SineSweep_10_6_length, 5:6]


    
    df_Vin_SineSweep_10_6_length = len(df_Vin_SineSweep_10_6)
    df_VinB_SineSweep_10_6 = df_Vin_SineSweep_10_6.iloc[0:df_Vin_SineSweep_10_6_length, :]

    df_Iin_SineSweep_10_6_length = len(df_Iin_SineSweep_10_6)
    df_IinB_SineSweep_10_6 = df_Iin_SineSweep_10_6.iloc[0:df_Iin_SineSweep_10_6_length, :]

    df_Vout_SineSweep_10_6_length = len(df_Vout_SineSweep_10_6)
    df_VoutB_SineSweep_10_6 = df_Vout_SineSweep_10_6.iloc[0:df_Vout_SineSweep_10_6_length, :]

    df_Iout_SineSweep_10_6_length = len(df_Iout_SineSweep_10_6)
    df_IoutB_SineSweep_10_6 = df_Iout_SineSweep_10_6.iloc[0:df_Iout_SineSweep_10_6_length, :]






    
    
    



    df_Pulse_5b_9 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In5b_Out9__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
   
    
    df_Pulse_5b_9_length = len(df_Pulse_5b_9)
    df_Vin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 2:3]
    df_Vout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 3:4]
    df_Iout_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 4:5]
    df_Iin_Pulse_5b_9 = df_Pulse_5b_9.iloc[0:df_Pulse_5b_9_length, 5:6]
    
    

    df_Vin_Pulse_5b_9_length = len(df_Vin_Pulse_5b_9)
    df_VinB_Pulse_5b_9 = df_Vin_Pulse_5b_9.iloc[0:df_Vin_Pulse_5b_9_length, :]
    
    df_Iin_Pulse_5b_9_length = len(df_Iin_Pulse_5b_9)
    df_IinB_Pulse_5b_9 = df_Iin_Pulse_5b_9.iloc[0:df_Iin_Pulse_5b_9_length, :]
    
    df_Vout_Pulse_5b_9_length = len(df_Vout_Pulse_5b_9)
    df_VoutB_Pulse_5b_9 = df_Vout_Pulse_5b_9.iloc[0:df_Vout_Pulse_5b_9_length, :]
    
    df_Iout_Pulse_5b_9_length = len(df_Iout_Pulse_5b_9)
    df_IoutB_Pulse_5b_9 = df_Iout_Pulse_5b_9.iloc[0:df_Iout_Pulse_5b_9_length, :]
    
    
    
    
    
    
    
    
    


    df_Pulse_6c_5 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__Pulse__In6c_Out5__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_Pulse_6c_5_length = len(df_Pulse_6c_5)
    df_Vin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 2:3]
    df_Vout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 3:4]
    df_Iout_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 4:5]
    df_Iin_Pulse_6c_5 = df_Pulse_6c_5.iloc[0:df_Pulse_6c_5_length, 5:6]
    
    
    df_Vin_Pulse_6c_5_length = len(df_Vin_Pulse_6c_5)
    df_VinB_Pulse_6c_5 = df_Vin_Pulse_6c_5.iloc[0:df_Vin_Pulse_6c_5_length, :]
    
    df_Iin_Pulse_6c_5_length = len(df_Iin_Pulse_6c_5)
    df_IinB_Pulse_6c_5 = df_Iin_Pulse_6c_5.iloc[0:df_Iin_Pulse_6c_5_length, :]
    
    df_Vout_Pulse_6c_5_length = len(df_Vout_Pulse_6c_5)
    df_VoutB_Pulse_6c_5 = df_Vout_Pulse_6c_5.iloc[0:df_Vout_Pulse_6c_5_length, :]
    
    df_Iout_Pulse_6c_5_length = len(df_Iout_Pulse_6c_5)
    df_IoutB_Pulse_6c_5 = df_Iout_Pulse_6c_5.iloc[0:df_Iout_Pulse_6c_5_length, :]
    
    
    
    
    
    
    
    
    
    
    
    
    from sklearn.preprocessing import MinMaxScaler


    scaler_Vin = MinMaxScaler()
    all_Vin_Pulse_6c_5 = scaler_Vin.fit_transform(df_VinB_Pulse_6c_5)


    scaler_Iout = MinMaxScaler()
    all_Iout_Pulse_5b_9 = scaler_Iout.fit_transform(df_IoutB_Pulse_5b_9)







    scaler_Vout = MinMaxScaler()
    all_Vout_Pulse_6c_5 = scaler_Vout.fit_transform(df_VoutB_Pulse_6c_5)


    scaler_Iin = MinMaxScaler()
    all_Iin_Pulse_6c_5 = scaler_Iin.fit_transform(df_IinB_Pulse_6c_5)
    
    
    
    
    
    
    all_Vin_SineSweep_10_6 = scaler_Vin.transform(df_VinB_SineSweep_10_6)
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    import torch.nn as nn
    import torch.optim as optim


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


           
            rin = torch.cat((a, b, c, d), 2)

            # Data is fed to the NN
            out = self.model['lin1'](rin)
           # print(out.shape)
            out = self.tanh(out)
            out, _ = self.model['lstm1'](out)
            out = self.model['lin2'](out)
            out = self.tanh(out)
            out, _ = self.model['lstm2'](out)
            out = self.model['lin3'](out)
 
            #print(out.shape)
            y_preda = out[:, :, -4].unsqueeze(-1)
            y_predb = out[:, :, -3].unsqueeze(-1)
            y_predc = out[:, :, -2].unsqueeze(-1)
            y_predd = out[:, :, -1].unsqueeze(-1)

            #print(y_pred.shape)
            #print(y_predb.shape)
            return y_preda, y_predb, y_predc, y_predd
        
        
        
        
        
        
        
    def generate_sequence(scaler_a, scaler_b, scaler_c, scaler_d, model, xa_sample, xb_sample, xc_sample, xd_sample):
        
        y_pred_tensor_a, y_pred_tensor_b, y_pred_tensor_c, y_pred_tensor_d = model(xa_sample, xb_sample, xc_sample, xd_sample)
        y_pred_red_a = y_pred_tensor_a[:, -1]
        y_pred_a = y_pred_red_a.cpu().tolist()
        y_pred_a = scaler_a.inverse_transform(y_pred_a)
        y_pred_red_b = y_pred_tensor_b[:, -1]
        y_pred_b = y_pred_red_b.cpu().tolist()
        y_pred_b = scaler_b.inverse_transform(y_pred_b) 
        y_pred_red_c = y_pred_tensor_c[:, -1]
        y_pred_c = y_pred_red_c.cpu().tolist()
        y_pred_c = scaler_c.inverse_transform(y_pred_c)
        y_pred_red_d = y_pred_tensor_d[:, -1]
        y_pred_d = y_pred_red_d.cpu().tolist()
        y_pred_d = scaler_d.inverse_transform(y_pred_d)
            
        return y_pred_a, y_pred_b, y_pred_c, y_pred_d
        
        
        
        
        
        
        
        
    model = MockupModel()

    #lossfn = nn.MSELoss(reduction='sum')

    learning_rate = 5e-3

   
    
    
    ADVar1_init = Variable(torch.zeros(500000, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True
    
    ADVar2_init = Variable(torch.zeros(500000, 1))
    ADVar2_scaled = scaler_Vout.transform(ADVar2_init)
    ADVar2_seq = transform_data(ADVar2_scaled, seq_len)
    ADVar2_seq.requires_grad = True

    #optimizer = optim.Adam([ADVar1_seq], lr=learning_rate)
    optimizer = optim.Adam([ADVar1_seq, ADVar2_seq], lr=learning_rate)




    
    checkpoint = torch.load(save_path_reconstruction_Iin_Vout_left, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #run = checkpoint['run']
    ADVar1_seq = checkpoint['ADVar1_seq']
    ADVar2_seq = checkpoint['ADVar2_seq']
    
    
    
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = ADVar1_seq.to(device)
    C_app_all = ADVar2_seq.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)





    start_time = time.time()
    ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6 = generate_sequence(scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, model, A_app_all, B_app_all, C_app_all, D_app_all)
    predTime = time.time() - start_time
    print("Calculation Time (generate sequence with cpu):", predTime)
    
    
    
    return ypredVin_SineSweep_10_6, ypredIin_SineSweep_10_6, ypredVout_SineSweep_10_6, ypredIout_SineSweep_10_6




