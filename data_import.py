# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:25:25 2023

@author: roche
"""




def load_and_transform_all_data(seq_len, device):
    import pandas as pd
    import numpy as np
    import torch
    from torch.autograd import Variable


    df_DC_2b_6 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__DC__In2b_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')


    df_DC_2b_6_length = len(df_DC_2b_6)
    df_Vin_DC_2b_6 = df_DC_2b_6.iloc[0:df_DC_2b_6_length, 2:3]
    df_Vout_DC_2b_6 = df_DC_2b_6.iloc[0:df_DC_2b_6_length, 3:4]
    df_Iout_DC_2b_6 = df_DC_2b_6.iloc[0:df_DC_2b_6_length, 4:5]
    df_Iin_DC_2b_6 = df_DC_2b_6.iloc[0:df_DC_2b_6_length, 5:6]
    
    
    
    df_Vin_DC_2b_6_length = len(df_Vin_DC_2b_6)
    df_VinB_DC_2b_6 = df_Vin_DC_2b_6.iloc[0:df_Vin_DC_2b_6_length, :]
    
    df_Iin_DC_2b_6_length = len(df_Iin_DC_2b_6)
    df_IinB_DC_2b_6 = df_Iin_DC_2b_6.iloc[0:df_Iin_DC_2b_6_length, :]
      
    df_Vout_DC_2b_6_length = len(df_Vout_DC_2b_6)  
    df_VoutB_DC_2b_6 = df_Vout_DC_2b_6.iloc[0:df_Vout_DC_2b_6_length, :]
    
    df_Iout_DC_2b_6_length = len(df_Iout_DC_2b_6)  
    df_IoutB_DC_2b_6 = df_Iout_DC_2b_6.iloc[0:df_Iout_DC_2b_6_length, :]
    
     
    
    df_train_end_DC_2b_6 = df_Vin_DC_2b_6_length*0.7
    df_val_start_DC_2b_6 = df_Vin_DC_2b_6_length*0.7 +1
    df_val_end_DC_2b_6 = df_Vin_DC_2b_6_length*0.9
    df_test_start_DC_2b_6 = df_Vin_DC_2b_6_length*0.9 +1
    
    
    
    
    df_train_Vin_DC_2b_6 = df_VinB_DC_2b_6[df_VinB_DC_2b_6.index < df_train_end_DC_2b_6]
    df_train_Iin_DC_2b_6 = df_IinB_DC_2b_6[df_IinB_DC_2b_6.index < df_train_end_DC_2b_6]
    df_train_Vout_DC_2b_6 = df_VoutB_DC_2b_6[df_VoutB_DC_2b_6.index < df_train_end_DC_2b_6]
    df_train_Iout_DC_2b_6 = df_IoutB_DC_2b_6[df_IoutB_DC_2b_6.index < df_train_end_DC_2b_6]
    
    df_val_Vin_DC_2b_6 = df_VinB_DC_2b_6[(df_VinB_DC_2b_6.index > df_val_start_DC_2b_6) & (df_VinB_DC_2b_6.index < df_val_end_DC_2b_6)]
    df_val_Iin_DC_2b_6 = df_IinB_DC_2b_6[(df_VinB_DC_2b_6.index > df_val_start_DC_2b_6) & (df_VinB_DC_2b_6.index < df_val_end_DC_2b_6)]
    df_val_Vout_DC_2b_6 = df_VoutB_DC_2b_6[(df_VinB_DC_2b_6.index > df_val_start_DC_2b_6) & (df_VinB_DC_2b_6.index < df_val_end_DC_2b_6)]
    df_val_Iout_DC_2b_6 = df_IoutB_DC_2b_6[(df_VinB_DC_2b_6.index > df_val_start_DC_2b_6) & (df_VinB_DC_2b_6.index < df_val_end_DC_2b_6)]
    
    df_test_Vin_DC_2b_6 = df_VinB_DC_2b_6[df_VinB_DC_2b_6.index > df_test_start_DC_2b_6]
    df_test_Iin_DC_2b_6 = df_IinB_DC_2b_6[df_VinB_DC_2b_6.index > df_test_start_DC_2b_6]
    df_test_Vout_DC_2b_6 = df_VoutB_DC_2b_6[df_VinB_DC_2b_6.index > df_test_start_DC_2b_6]
    df_test_Iout_DC_2b_6 = df_IoutB_DC_2b_6[df_VinB_DC_2b_6.index > df_test_start_DC_2b_6]
    
    
    
 
    
 
    
 
    df_DC_2c_8 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__DC__In2c_Out8__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_DC_2c_8_length = len(df_DC_2c_8)
    df_Vin_DC_2c_8 = df_DC_2c_8.iloc[0:df_DC_2c_8_length, 2:3]
    df_Vout_DC_2c_8 = df_DC_2c_8.iloc[0:df_DC_2c_8_length, 3:4]
    df_Iout_DC_2c_8 = df_DC_2c_8.iloc[0:df_DC_2c_8_length, 4:5]
    df_Iin_DC_2c_8 = df_DC_2c_8.iloc[0:df_DC_2c_8_length, 5:6]
    
        
    df_Vin_DC_2c_8_length = len(df_Vin_DC_2c_8)    
    df_VinB_DC_2c_8 = df_Vin_DC_2c_8.iloc[0:df_Vin_DC_2c_8_length, :]
        
    df_Iin_DC_2c_8_length = len(df_Iin_DC_2c_8)   
    df_IinB_DC_2c_8 = df_Iin_DC_2c_8.iloc[0:df_Iin_DC_2c_8_length, :]
    
    df_Vout_DC_2c_8_length = len(df_Vout_DC_2c_8)    
    df_VoutB_DC_2c_8 = df_Vout_DC_2c_8.iloc[0:df_Vout_DC_2c_8_length, :]
      
    df_Iout_DC_2c_8_length = len(df_Iout_DC_2c_8)
    df_IoutB_DC_2c_8 = df_Iout_DC_2c_8.iloc[0:df_Iout_DC_2c_8_length, :]
    
    
     
    df_train_end_DC_2c_8 = df_Vin_DC_2c_8_length*0.7
    df_val_start_DC_2c_8 = df_Vin_DC_2c_8_length*0.7 +1
    df_val_end_DC_2c_8 = df_Vin_DC_2c_8_length*0.9
    df_test_start_DC_2c_8 = df_Vin_DC_2c_8_length*0.9 +1
    
    
    
    
    
    df_train_Vin_DC_2c_8 = df_VinB_DC_2c_8[df_VinB_DC_2c_8.index < df_train_end_DC_2c_8]
    df_train_Iin_DC_2c_8 = df_IinB_DC_2c_8[df_IinB_DC_2c_8.index < df_train_end_DC_2c_8]
    df_train_Vout_DC_2c_8 = df_VoutB_DC_2c_8[df_VoutB_DC_2c_8.index < df_train_end_DC_2c_8]
    df_train_Iout_DC_2c_8 = df_IoutB_DC_2c_8[df_IoutB_DC_2c_8.index < df_train_end_DC_2c_8]
    
    df_val_Vin_DC_2c_8 = df_VinB_DC_2c_8[(df_VinB_DC_2c_8.index > df_val_start_DC_2c_8) & (df_VinB_DC_2c_8.index < df_val_end_DC_2c_8)]
    df_val_Iin_DC_2c_8 = df_IinB_DC_2c_8[(df_VinB_DC_2c_8.index > df_val_start_DC_2c_8) & (df_VinB_DC_2c_8.index < df_val_end_DC_2c_8)]
    df_val_Vout_DC_2c_8 = df_VoutB_DC_2c_8[(df_VinB_DC_2c_8.index > df_val_start_DC_2c_8) & (df_VinB_DC_2c_8.index < df_val_end_DC_2c_8)]
    df_val_Iout_DC_2c_8 = df_IoutB_DC_2c_8[(df_VinB_DC_2c_8.index > df_val_start_DC_2c_8) & (df_VinB_DC_2c_8.index < df_val_end_DC_2c_8)]
    
    df_test_Vin_DC_2c_8 = df_VinB_DC_2c_8[df_VinB_DC_2c_8.index > df_test_start_DC_2c_8]
    df_test_Iin_DC_2c_8 = df_IinB_DC_2c_8[df_VinB_DC_2c_8.index > df_test_start_DC_2c_8]
    df_test_Vout_DC_2c_8 = df_VoutB_DC_2c_8[df_VinB_DC_2c_8.index > df_test_start_DC_2c_8]
    df_test_Iout_DC_2c_8 = df_IoutB_DC_2c_8[df_VinB_DC_2c_8.index > df_test_start_DC_2c_8]
    
    
    
    
    
    df_DC_3a_6 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__DC__In3a_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_DC_3a_6_length = len(df_DC_3a_6)
    df_Vin_DC_3a_6 = df_DC_3a_6.iloc[0:df_DC_3a_6_length, 2:3]
    df_Vout_DC_3a_6 = df_DC_3a_6.iloc[0:df_DC_3a_6_length, 3:4]
    df_Iout_DC_3a_6 = df_DC_3a_6.iloc[0:df_DC_3a_6_length, 4:5]
    df_Iin_DC_3a_6 = df_DC_3a_6.iloc[0:df_DC_3a_6_length, 5:6]
    
    
    
    df_Vin_DC_3a_6_length = len(df_Vin_DC_3a_6)    
    df_VinB_DC_3a_6 = df_Vin_DC_3a_6.iloc[0:df_Vin_DC_3a_6_length, :]
        
    df_Iin_DC_3a_6_length = len(df_Iin_DC_3a_6)   
    df_IinB_DC_3a_6 = df_Iin_DC_3a_6.iloc[0:df_Iin_DC_3a_6_length, :]
    
    df_Vout_DC_3a_6_length = len(df_Vout_DC_3a_6)    
    df_VoutB_DC_3a_6 = df_Vout_DC_3a_6.iloc[0:df_Vout_DC_3a_6_length, :]
    
    df_Iout_DC_3a_6_length = len(df_Iout_DC_3a_6)   
    df_IoutB_DC_3a_6 = df_Iout_DC_3a_6.iloc[0:df_Iout_DC_3a_6_length, :]
    
    

    df_train_end_DC_3a_6 = df_Vin_DC_3a_6_length*0.7
    df_val_start_DC_3a_6 = df_Vin_DC_3a_6_length*0.7 +1
    df_val_end_DC_3a_6 = df_Vin_DC_3a_6_length*0.9
    df_test_start_DC_3a_6 = df_Vin_DC_3a_6_length*0.9 +1
    
    
    
    
    
    df_train_Vin_DC_3a_6 = df_VinB_DC_3a_6[df_VinB_DC_3a_6.index < df_train_end_DC_3a_6]
    df_train_Iin_DC_3a_6 = df_IinB_DC_3a_6[df_IinB_DC_3a_6.index < df_train_end_DC_3a_6]
    df_train_Vout_DC_3a_6 = df_VoutB_DC_3a_6[df_VoutB_DC_3a_6.index < df_train_end_DC_3a_6]
    df_train_Iout_DC_3a_6 = df_IoutB_DC_3a_6[df_IoutB_DC_3a_6.index < df_train_end_DC_3a_6]
    
    df_val_Vin_DC_3a_6 = df_VinB_DC_3a_6[(df_VinB_DC_3a_6.index > df_val_start_DC_3a_6) & (df_VinB_DC_3a_6.index < df_val_end_DC_3a_6)]
    df_val_Iin_DC_3a_6 = df_IinB_DC_3a_6[(df_VinB_DC_3a_6.index > df_val_start_DC_3a_6) & (df_VinB_DC_3a_6.index < df_val_end_DC_3a_6)]
    df_val_Vout_DC_3a_6 = df_VoutB_DC_3a_6[(df_VinB_DC_3a_6.index > df_val_start_DC_3a_6) & (df_VinB_DC_3a_6.index < df_val_end_DC_3a_6)]
    df_val_Iout_DC_3a_6 = df_IoutB_DC_3a_6[(df_VinB_DC_3a_6.index > df_val_start_DC_3a_6) & (df_VinB_DC_3a_6.index < df_val_end_DC_3a_6)]
    
    df_test_Vin_DC_3a_6 = df_VinB_DC_3a_6[df_VinB_DC_3a_6.index > df_test_start_DC_3a_6]
    df_test_Iin_DC_3a_6 = df_IinB_DC_3a_6[df_VinB_DC_3a_6.index > df_test_start_DC_3a_6]
    df_test_Vout_DC_3a_6 = df_VoutB_DC_3a_6[df_VinB_DC_3a_6.index > df_test_start_DC_3a_6]
    df_test_Iout_DC_3a_6 = df_IoutB_DC_3a_6[df_VinB_DC_3a_6.index > df_test_start_DC_3a_6]
    
    
    
    
    
    
    
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
    
    
    df_train_end_Pulse_5b_9 = df_Vin_Pulse_5b_9_length*0.7
    df_val_start_Pulse_5b_9 = df_Vin_Pulse_5b_9_length*0.7 +1
    df_val_end_Pulse_5b_9 = df_Vin_Pulse_5b_9_length*0.9
    df_test_start_Pulse_5b_9 = df_Vin_Pulse_5b_9_length*0.9 +1
    
    
    
    
    df_train_Vin_Pulse_5b_9 = df_VinB_Pulse_5b_9[df_VinB_Pulse_5b_9.index < df_train_end_Pulse_5b_9]
    df_train_Iin_Pulse_5b_9 = df_IinB_Pulse_5b_9[df_IinB_Pulse_5b_9.index < df_train_end_Pulse_5b_9]
    df_train_Vout_Pulse_5b_9 = df_VoutB_Pulse_5b_9[df_VoutB_Pulse_5b_9.index < df_train_end_Pulse_5b_9]
    df_train_Iout_Pulse_5b_9 = df_IoutB_Pulse_5b_9[df_IoutB_Pulse_5b_9.index < df_train_end_Pulse_5b_9]
    
    df_val_Vin_Pulse_5b_9 = df_VinB_Pulse_5b_9[(df_VinB_Pulse_5b_9.index > df_val_start_Pulse_5b_9) & (df_VinB_Pulse_5b_9.index < df_val_end_Pulse_5b_9)]
    df_val_Iin_Pulse_5b_9 = df_IinB_Pulse_5b_9[(df_VinB_Pulse_5b_9.index > df_val_start_Pulse_5b_9) & (df_VinB_Pulse_5b_9.index < df_val_end_Pulse_5b_9)]
    df_val_Vout_Pulse_5b_9 = df_VoutB_Pulse_5b_9[(df_VinB_Pulse_5b_9.index > df_val_start_Pulse_5b_9) & (df_VinB_Pulse_5b_9.index < df_val_end_Pulse_5b_9)]
    df_val_Iout_Pulse_5b_9 = df_IoutB_Pulse_5b_9[(df_VinB_Pulse_5b_9.index > df_val_start_Pulse_5b_9) & (df_VinB_Pulse_5b_9.index < df_val_end_Pulse_5b_9)]
    
    df_test_Vin_Pulse_5b_9 = df_VinB_Pulse_5b_9[df_VinB_Pulse_5b_9.index > df_test_start_Pulse_5b_9]
    df_test_Iin_Pulse_5b_9 = df_IinB_Pulse_5b_9[df_VinB_Pulse_5b_9.index > df_test_start_Pulse_5b_9]
    df_test_Vout_Pulse_5b_9 = df_VoutB_Pulse_5b_9[df_VinB_Pulse_5b_9.index > df_test_start_Pulse_5b_9]
    df_test_Iout_Pulse_5b_9 = df_IoutB_Pulse_5b_9[df_VinB_Pulse_5b_9.index > df_test_start_Pulse_5b_9]
    
    
    
    
    
    
    
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
    
    
    

    df_train_end_Pulse_6c_5 = df_Vin_Pulse_6c_5_length*0.7
    df_val_start_Pulse_6c_5 = df_Vin_Pulse_6c_5_length*0.7 +1
    df_val_end_Pulse_6c_5 = df_Vin_Pulse_6c_5_length*0.9
    df_test_start_Pulse_6c_5 = df_Vin_Pulse_6c_5_length*0.9 +1
    
    
    
    
    
    df_train_Vin_Pulse_6c_5 = df_VinB_Pulse_6c_5[df_VinB_Pulse_6c_5.index < df_train_end_Pulse_6c_5]
    df_train_Iin_Pulse_6c_5 = df_IinB_Pulse_6c_5[df_IinB_Pulse_6c_5.index < df_train_end_Pulse_6c_5]
    df_train_Vout_Pulse_6c_5 = df_VoutB_Pulse_6c_5[df_VoutB_Pulse_6c_5.index < df_train_end_Pulse_6c_5]
    df_train_Iout_Pulse_6c_5 = df_IoutB_Pulse_6c_5[df_IoutB_Pulse_6c_5.index < df_train_end_Pulse_6c_5]
    
    df_val_Vin_Pulse_6c_5 = df_VinB_Pulse_6c_5[(df_VinB_Pulse_6c_5.index > df_val_start_Pulse_6c_5) & (df_VinB_Pulse_6c_5.index < df_val_end_Pulse_6c_5)]
    df_val_Iin_Pulse_6c_5 = df_IinB_Pulse_6c_5[(df_VinB_Pulse_6c_5.index > df_val_start_Pulse_6c_5) & (df_VinB_Pulse_6c_5.index < df_val_end_Pulse_6c_5)]
    df_val_Vout_Pulse_6c_5 = df_VoutB_Pulse_6c_5[(df_VinB_Pulse_6c_5.index > df_val_start_Pulse_6c_5) & (df_VinB_Pulse_6c_5.index < df_val_end_Pulse_6c_5)]
    df_val_Iout_Pulse_6c_5 = df_IoutB_Pulse_6c_5[(df_VinB_Pulse_6c_5.index > df_val_start_Pulse_6c_5) & (df_VinB_Pulse_6c_5.index < df_val_end_Pulse_6c_5)]
    
    df_test_Vin_Pulse_6c_5 = df_VinB_Pulse_6c_5[df_VinB_Pulse_6c_5.index > df_test_start_Pulse_6c_5]
    df_test_Iin_Pulse_6c_5 = df_IinB_Pulse_6c_5[df_VinB_Pulse_6c_5.index > df_test_start_Pulse_6c_5]
    df_test_Vout_Pulse_6c_5 = df_VoutB_Pulse_6c_5[df_VinB_Pulse_6c_5.index > df_test_start_Pulse_6c_5]
    df_test_Iout_Pulse_6c_5 = df_IoutB_Pulse_6c_5[df_VinB_Pulse_6c_5.index > df_test_start_Pulse_6c_5]
    
    
    
    
    
    
    df_PWM3 = pd.read_csv('data/AE1_x10/train/RLC__C_Charge__PWM3__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_PWM3_length = len(df_PWM3)
    df_Vin_PWM3 = df_PWM3.iloc[0:df_PWM3_length, 2:3]
    df_Vout_PWM3 = df_PWM3.iloc[0:df_PWM3_length, 3:4]
    df_Iout_PWM3 = df_PWM3.iloc[0:df_PWM3_length, 4:5]
    df_Iin_PWM3 = df_PWM3.iloc[0:df_PWM3_length, 5:6]
    
    
    df_Vin_PWM3_length = len(df_Vin_PWM3)    
    df_VinB_PWM3 = df_Vin_PWM3.iloc[0:df_Vin_PWM3_length, :]
    
    df_Iin_PWM3_length = len(df_Iin_PWM3)    
    df_IinB_PWM3 = df_Iin_PWM3.iloc[0:df_Iin_PWM3_length, :]
       
    df_Vout_PWM3_length = len(df_Vout_PWM3)    
    df_VoutB_PWM3 = df_Vout_PWM3.iloc[0:df_Vout_PWM3_length, :]
       
    df_Iout_PWM3_length = len(df_Iout_PWM3)   
    df_IoutB_PWM3 = df_Iout_PWM3.iloc[0:df_Iout_PWM3_length, :]
    
    
    
    df_train_end_PWM3 = df_Vin_PWM3_length*0.7
    df_val_start_PWM3 = df_Vin_PWM3_length*0.7 +1
    df_val_end_PWM3 = df_Vin_PWM3_length*0.9
    df_test_start_PWM3 = df_Vin_PWM3_length*0.9 +1
    
    
    
    df_train_Vin_PWM3 = df_VinB_PWM3[df_VinB_PWM3.index < df_train_end_PWM3]
    df_train_Iin_PWM3 = df_IinB_PWM3[df_IinB_PWM3.index < df_train_end_PWM3]
    df_train_Vout_PWM3 = df_VoutB_PWM3[df_VoutB_PWM3.index < df_train_end_PWM3]
    df_train_Iout_PWM3 = df_IoutB_PWM3[df_IoutB_PWM3.index < df_train_end_PWM3]
    
    df_val_Vin_PWM3 = df_VinB_PWM3[(df_VinB_PWM3.index > df_val_start_PWM3) & (df_VinB_PWM3.index < df_val_end_PWM3)]
    df_val_Iin_PWM3 = df_IinB_PWM3[(df_VinB_PWM3.index > df_val_start_PWM3) & (df_VinB_PWM3.index < df_val_end_PWM3)]
    df_val_Vout_PWM3 = df_VoutB_PWM3[(df_VinB_PWM3.index > df_val_start_PWM3) & (df_VinB_PWM3.index < df_val_end_PWM3)]
    df_val_Iout_PWM3 = df_IoutB_PWM3[(df_VinB_PWM3.index > df_val_start_PWM3) & (df_VinB_PWM3.index < df_val_end_PWM3)]
    
    df_test_Vin_PWM3 = df_VinB_PWM3[df_VinB_PWM3.index > df_test_start_PWM3]
    df_test_Iin_PWM3 = df_IinB_PWM3[df_VinB_PWM3.index > df_test_start_PWM3]
    df_test_Vout_PWM3 = df_VoutB_PWM3[df_VinB_PWM3.index > df_test_start_PWM3]
    df_test_Iout_PWM3 = df_IoutB_PWM3[df_VinB_PWM3.index > df_test_start_PWM3]
    
    
    
    
    
    
    
    df_DC_1b_6 = pd.read_csv('data/AE1_x10/test/RLC__C_Charge__DC__In1b_Out6__5ms_resampled_500000p.txt.gz', delimiter=",", compression='gzip')
    
    
    df_DC_1b_6_length = len(df_DC_1b_6)
    df_Vin_DC_1b_6 = df_DC_1b_6.iloc[0:df_DC_1b_6_length, 2:3]
    df_Vout_DC_1b_6 = df_DC_1b_6.iloc[0:df_DC_1b_6_length, 3:4]
    df_Iout_DC_1b_6 = df_DC_1b_6.iloc[0:df_DC_1b_6_length, 4:5]
    df_Iin_DC_1b_6 = df_DC_1b_6.iloc[0:df_DC_1b_6_length, 5:6]
    
    
    df_Vin_DC_1b_6_length = len(df_Vin_DC_1b_6) 
    df_VinB_DC_1b_6 = df_Vin_DC_1b_6.iloc[0:df_Vin_DC_1b_6_length, :]
    
    df_Iin_DC_1b_6_length = len(df_Iin_DC_1b_6)   
    df_IinB_DC_1b_6 = df_Iin_DC_1b_6.iloc[0:df_Iin_DC_1b_6_length, :]
    
    df_Vout_DC_1b_6_length = len(df_Vout_DC_1b_6) 
    df_VoutB_DC_1b_6 = df_Vout_DC_1b_6.iloc[0:df_Vout_DC_1b_6_length, :]
    
    df_Iout_DC_1b_6_length = len(df_Iout_DC_1b_6)
    df_IoutB_DC_1b_6 = df_Iout_DC_1b_6.iloc[0:df_Iout_DC_1b_6_length, :]
    
    

    df_train_end_DC_1b_6 = df_Vin_DC_1b_6_length*0.7
    df_val_start_DC_1b_6 = df_Vin_DC_1b_6_length*0.7 +1
    df_val_end_DC_1b_6 = df_Vin_DC_1b_6_length*0.9
    df_test_start_DC_1b_6 = df_Vin_DC_1b_6_length*0.9 +1
    
    
    
    
    
    df_train_Vin_DC_1b_6 = df_VinB_DC_1b_6[df_VinB_DC_1b_6.index < df_train_end_DC_1b_6]
    df_train_Iin_DC_1b_6 = df_IinB_DC_1b_6[df_IinB_DC_1b_6.index < df_train_end_DC_1b_6]
    df_train_Vout_DC_1b_6 = df_VoutB_DC_1b_6[df_VoutB_DC_1b_6.index < df_train_end_DC_1b_6]
    df_train_Iout_DC_1b_6 = df_IoutB_DC_1b_6[df_IoutB_DC_1b_6.index < df_train_end_DC_1b_6]
    
    df_val_Vin_DC_1b_6 = df_VinB_DC_1b_6[(df_VinB_DC_1b_6.index > df_val_start_DC_1b_6) & (df_VinB_DC_1b_6.index < df_val_end_DC_1b_6)]
    df_val_Iin_DC_1b_6 = df_IinB_DC_1b_6[(df_VinB_DC_1b_6.index > df_val_start_DC_1b_6) & (df_VinB_DC_1b_6.index < df_val_end_DC_1b_6)]
    df_val_Vout_DC_1b_6 = df_VoutB_DC_1b_6[(df_VinB_DC_1b_6.index > df_val_start_DC_1b_6) & (df_VinB_DC_1b_6.index < df_val_end_DC_1b_6)]
    df_val_Iout_DC_1b_6 = df_IoutB_DC_1b_6[(df_VinB_DC_1b_6.index > df_val_start_DC_1b_6) & (df_VinB_DC_1b_6.index < df_val_end_DC_1b_6)]
    
    df_test_Vin_DC_1b_6 = df_VinB_DC_1b_6[df_VinB_DC_1b_6.index > df_test_start_DC_1b_6]
    df_test_Iin_DC_1b_6 = df_IinB_DC_1b_6[df_VinB_DC_1b_6.index > df_test_start_DC_1b_6]
    df_test_Vout_DC_1b_6 = df_VoutB_DC_1b_6[df_VinB_DC_1b_6.index > df_test_start_DC_1b_6]
    df_test_Iout_DC_1b_6 = df_IoutB_DC_1b_6[df_VinB_DC_1b_6.index > df_test_start_DC_1b_6]
    
    
    
    
    
    
    
    
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
    
    
    
    
    df_train_end_SineSweep_10_6 = df_Vin_SineSweep_10_6_length*0.7
    df_val_start_SineSweep_10_6 = df_Vin_SineSweep_10_6_length*0.7 +1
    df_val_end_SineSweep_10_6 = df_Vin_SineSweep_10_6_length*0.9
    df_test_start_SineSweep_10_6 = df_Vin_SineSweep_10_6_length*0.9 +1
    
    
    
    
    
    df_train_Vin_SineSweep_10_6 = df_VinB_SineSweep_10_6[df_VinB_SineSweep_10_6.index < df_train_end_SineSweep_10_6]
    df_train_Iin_SineSweep_10_6 = df_IinB_SineSweep_10_6[df_IinB_SineSweep_10_6.index < df_train_end_SineSweep_10_6]
    df_train_Vout_SineSweep_10_6 = df_VoutB_SineSweep_10_6[df_VoutB_SineSweep_10_6.index < df_train_end_SineSweep_10_6]
    df_train_Iout_SineSweep_10_6 = df_IoutB_SineSweep_10_6[df_IoutB_SineSweep_10_6.index < df_train_end_SineSweep_10_6]
    
    df_val_Vin_SineSweep_10_6 = df_VinB_SineSweep_10_6[(df_VinB_SineSweep_10_6.index > df_val_start_SineSweep_10_6) & (df_VinB_SineSweep_10_6.index < df_val_end_SineSweep_10_6)]
    df_val_Iin_SineSweep_10_6 = df_IinB_SineSweep_10_6[(df_VinB_SineSweep_10_6.index > df_val_start_SineSweep_10_6) & (df_VinB_SineSweep_10_6.index < df_val_end_SineSweep_10_6)]
    df_val_Vout_SineSweep_10_6 = df_VoutB_SineSweep_10_6[(df_VinB_SineSweep_10_6.index > df_val_start_SineSweep_10_6) & (df_VinB_SineSweep_10_6.index < df_val_end_SineSweep_10_6)]
    df_val_Iout_SineSweep_10_6 = df_IoutB_SineSweep_10_6[(df_VinB_SineSweep_10_6.index > df_val_start_SineSweep_10_6) & (df_VinB_SineSweep_10_6.index < df_val_end_SineSweep_10_6)]
    
    df_test_Vin_SineSweep_10_6 = df_VinB_SineSweep_10_6[df_VinB_SineSweep_10_6.index > df_test_start_SineSweep_10_6]
    df_test_Iin_SineSweep_10_6 = df_IinB_SineSweep_10_6[df_VinB_SineSweep_10_6.index > df_test_start_SineSweep_10_6]
    df_test_Vout_SineSweep_10_6 = df_VoutB_SineSweep_10_6[df_VinB_SineSweep_10_6.index > df_test_start_SineSweep_10_6]
    df_test_Iout_SineSweep_10_6 = df_IoutB_SineSweep_10_6[df_VinB_SineSweep_10_6.index > df_test_start_SineSweep_10_6]
    
    
   
























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
    Vin_train_arr_SineSweep_10_6 = scaler_Vin.transform(df_train_Vin_SineSweep_10_6)
    Vin_val_arr_SineSweep_10_6 = scaler_Vin.transform(df_val_Vin_SineSweep_10_6)
    Vin_test_arr_SineSweep_10_6 = scaler_Vin.transform(df_test_Vin_SineSweep_10_6)
    
    

    
    all_Iin_SineSweep_10_6 = scaler_Iin.transform(df_IinB_SineSweep_10_6)
    Iin_train_arr_SineSweep_10_6 = scaler_Iin.transform(df_train_Iin_SineSweep_10_6)
    Iin_val_arr_SineSweep_10_6 = scaler_Iin.transform(df_val_Iin_SineSweep_10_6)
    Iin_test_arr_SineSweep_10_6 = scaler_Iin.transform(df_test_Iin_SineSweep_10_6)
    
    
    

    
    all_Vout_SineSweep_10_6 = scaler_Vout.transform(df_VoutB_SineSweep_10_6)
    Vout_train_arr_SineSweep_10_6 = scaler_Vout.transform(df_train_Vout_SineSweep_10_6)
    Vout_val_arr_SineSweep_10_6 = scaler_Vout.transform(df_val_Vout_SineSweep_10_6)
    Vout_test_arr_SineSweep_10_6 = scaler_Vout.transform(df_test_Vout_SineSweep_10_6)
    
    

    
    all_Iout_SineSweep_10_6 = scaler_Iout.transform(df_IoutB_SineSweep_10_6)
    Iout_train_arr_SineSweep_10_6 = scaler_Iout.transform(df_train_Iout_SineSweep_10_6)
    Iout_val_arr_SineSweep_10_6 = scaler_Iout.transform(df_val_Iout_SineSweep_10_6)
    Iout_test_arr_SineSweep_10_6 = scaler_Iout.transform(df_test_Iout_SineSweep_10_6)
    
    
    
    
    
    
    
    
    
    
    
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var
    
    
    
    
    
    
    
    
    
    
    
    Vin_train_SineSweep_10_6 = transform_data(Vin_train_arr_SineSweep_10_6, seq_len)
    Vout_train_SineSweep_10_6 = transform_data(Vout_train_arr_SineSweep_10_6, seq_len)
    
    Vin_val_SineSweep_10_6 = transform_data(Vin_val_arr_SineSweep_10_6, seq_len)
    Vout_val_SineSweep_10_6 = transform_data(Vout_val_arr_SineSweep_10_6, seq_len)
    
    Vin_test_SineSweep_10_6 = transform_data(Vin_test_arr_SineSweep_10_6, seq_len)
    Vout_test_SineSweep_10_6 = transform_data(Vout_test_arr_SineSweep_10_6, seq_len)
    
    
    Vin_all_SineSweep_10_6 = transform_data(all_Vin_SineSweep_10_6, seq_len)
    Vout_all_SineSweep_10_6 = transform_data(all_Vout_SineSweep_10_6, seq_len)
    
    
    
    
    Iin_train_SineSweep_10_6 = transform_data(Iin_train_arr_SineSweep_10_6, seq_len)
    Iout_train_SineSweep_10_6 = transform_data(Iout_train_arr_SineSweep_10_6, seq_len)
    
    Iin_val_SineSweep_10_6 = transform_data(Iin_val_arr_SineSweep_10_6, seq_len)
    Iout_val_SineSweep_10_6 = transform_data(Iout_val_arr_SineSweep_10_6, seq_len)
    
    Iin_test_SineSweep_10_6 = transform_data(Iin_test_arr_SineSweep_10_6, seq_len)
    Iout_test_SineSweep_10_6 = transform_data(Iout_test_arr_SineSweep_10_6, seq_len)
    
    
    Iin_all_SineSweep_10_6 = transform_data(all_Iin_SineSweep_10_6, seq_len)
    Iout_all_SineSweep_10_6 = transform_data(all_Iout_SineSweep_10_6, seq_len)
    
    
    
    
    
    
    
    
    

    
    all_Vin_DC_2b_6 = scaler_Vin.transform(df_VinB_DC_2b_6)
    Vin_train_arr_DC_2b_6 = scaler_Vin.transform(df_train_Vin_DC_2b_6)
    Vin_val_arr_DC_2b_6 = scaler_Vin.transform(df_val_Vin_DC_2b_6)
    Vin_test_arr_DC_2b_6 = scaler_Vin.transform(df_test_Vin_DC_2b_6)
    
    

    
    all_Iin_DC_2b_6 = scaler_Iin.transform(df_IinB_DC_2b_6)
    Iin_train_arr_DC_2b_6 = scaler_Iin.transform(df_train_Iin_DC_2b_6)
    Iin_val_arr_DC_2b_6 = scaler_Iin.transform(df_val_Iin_DC_2b_6)
    Iin_test_arr_DC_2b_6 = scaler_Iin.transform(df_test_Iin_DC_2b_6)
    
    
    
 
    
    all_Vout_DC_2b_6 = scaler_Vout.transform(df_VoutB_DC_2b_6)
    Vout_train_arr_DC_2b_6 = scaler_Vout.transform(df_train_Vout_DC_2b_6)
    Vout_val_arr_DC_2b_6 = scaler_Vout.transform(df_val_Vout_DC_2b_6)
    Vout_test_arr_DC_2b_6 = scaler_Vout.transform(df_test_Vout_DC_2b_6)
    
    

    
    all_Iout_DC_2b_6 = scaler_Iout.transform(df_IoutB_DC_2b_6)
    Iout_train_arr_DC_2b_6 = scaler_Iout.transform(df_train_Iout_DC_2b_6)
    Iout_val_arr_DC_2b_6 = scaler_Iout.transform(df_val_Iout_DC_2b_6)
    Iout_test_arr_DC_2b_6 = scaler_Iout.transform(df_test_Iout_DC_2b_6)
    
    
    
    
    Vin_train_DC_2b_6 = transform_data(Vin_train_arr_DC_2b_6, seq_len)
    Vout_train_DC_2b_6 = transform_data(Vout_train_arr_DC_2b_6, seq_len)
    
    Vin_val_DC_2b_6 = transform_data(Vin_val_arr_DC_2b_6, seq_len)
    Vout_val_DC_2b_6 = transform_data(Vout_val_arr_DC_2b_6, seq_len)
    
    Vin_test_DC_2b_6 = transform_data(Vin_test_arr_DC_2b_6, seq_len)
    Vout_test_DC_2b_6 = transform_data(Vout_test_arr_DC_2b_6, seq_len)
    
    
    Vin_all_DC_2b_6 = transform_data(all_Vin_DC_2b_6, seq_len)
    Vout_all_DC_2b_6 = transform_data(all_Vout_DC_2b_6, seq_len)
    
    
    
    
    Iin_train_DC_2b_6 = transform_data(Iin_train_arr_DC_2b_6, seq_len)
    Iout_train_DC_2b_6 = transform_data(Iout_train_arr_DC_2b_6, seq_len)
    
    Iin_val_DC_2b_6 = transform_data(Iin_val_arr_DC_2b_6, seq_len)
    Iout_val_DC_2b_6 = transform_data(Iout_val_arr_DC_2b_6, seq_len)
    
    Iin_test_DC_2b_6 = transform_data(Iin_test_arr_DC_2b_6, seq_len)
    Iout_test_DC_2b_6 = transform_data(Iout_test_arr_DC_2b_6, seq_len)
    
    
    Iin_all_DC_2b_6 = transform_data(all_Iin_DC_2b_6, seq_len)
    Iout_all_DC_2b_6 = transform_data(all_Iout_DC_2b_6, seq_len)
    
    
    
    
    
    
    
    

    
    all_Vin_DC_2c_8 = scaler_Vin.transform(df_VinB_DC_2c_8)
    Vin_train_arr_DC_2c_8 = scaler_Vin.transform(df_train_Vin_DC_2c_8)
    Vin_val_arr_DC_2c_8 = scaler_Vin.transform(df_val_Vin_DC_2c_8)
    Vin_test_arr_DC_2c_8 = scaler_Vin.transform(df_test_Vin_DC_2c_8)
    
    

    
    all_Iin_DC_2c_8 = scaler_Iin.transform(df_IinB_DC_2c_8)
    Iin_train_arr_DC_2c_8 = scaler_Iin.transform(df_train_Iin_DC_2c_8)
    Iin_val_arr_DC_2c_8 = scaler_Iin.transform(df_val_Iin_DC_2c_8)
    Iin_test_arr_DC_2c_8 = scaler_Iin.transform(df_test_Iin_DC_2c_8)
    
    
    

    
    all_Vout_DC_2c_8 = scaler_Vout.transform(df_VoutB_DC_2c_8)
    Vout_train_arr_DC_2c_8 = scaler_Vout.transform(df_train_Vout_DC_2c_8)
    Vout_val_arr_DC_2c_8 = scaler_Vout.transform(df_val_Vout_DC_2c_8)
    Vout_test_arr_DC_2c_8 = scaler_Vout.transform(df_test_Vout_DC_2c_8)
    
    

    
    all_Iout_DC_2c_8 = scaler_Iout.transform(df_IoutB_DC_2c_8)
    Iout_train_arr_DC_2c_8 = scaler_Iout.transform(df_train_Iout_DC_2c_8)
    Iout_val_arr_DC_2c_8 = scaler_Iout.transform(df_val_Iout_DC_2c_8)
    Iout_test_arr_DC_2c_8 = scaler_Iout.transform(df_test_Iout_DC_2c_8)
    
    
    
    
    
    
    
    
    
    Vin_train_DC_2c_8 = transform_data(Vin_train_arr_DC_2c_8, seq_len)
    Vout_train_DC_2c_8 = transform_data(Vout_train_arr_DC_2c_8, seq_len)
    
    Vin_val_DC_2c_8 = transform_data(Vin_val_arr_DC_2c_8, seq_len)
    Vout_val_DC_2c_8 = transform_data(Vout_val_arr_DC_2c_8, seq_len)
    
    Vin_test_DC_2c_8 = transform_data(Vin_test_arr_DC_2c_8, seq_len)
    Vout_test_DC_2c_8 = transform_data(Vout_test_arr_DC_2c_8, seq_len)
    
    
    Vin_all_DC_2c_8 = transform_data(all_Vin_DC_2c_8, seq_len)
    Vout_all_DC_2c_8 = transform_data(all_Vout_DC_2c_8, seq_len)
    
    
    
    
    Iin_train_DC_2c_8 = transform_data(Iin_train_arr_DC_2c_8, seq_len)
    Iout_train_DC_2c_8 = transform_data(Iout_train_arr_DC_2c_8, seq_len)
    
    Iin_val_DC_2c_8 = transform_data(Iin_val_arr_DC_2c_8, seq_len)
    Iout_val_DC_2c_8 = transform_data(Iout_val_arr_DC_2c_8, seq_len)
    
    Iin_test_DC_2c_8 = transform_data(Iin_test_arr_DC_2c_8, seq_len)
    Iout_test_DC_2c_8 = transform_data(Iout_test_arr_DC_2c_8, seq_len)
    
    
    Iin_all_DC_2c_8 = transform_data(all_Iin_DC_2c_8, seq_len)
    Iout_all_DC_2c_8 = transform_data(all_Iout_DC_2c_8, seq_len)
    
    
    
    
    
    
    
    
    
    

    
    all_Vin_DC_3a_6 = scaler_Vin.transform(df_VinB_DC_3a_6)
    Vin_train_arr_DC_3a_6 = scaler_Vin.transform(df_train_Vin_DC_3a_6)
    Vin_val_arr_DC_3a_6 = scaler_Vin.transform(df_val_Vin_DC_3a_6)
    Vin_test_arr_DC_3a_6 = scaler_Vin.transform(df_test_Vin_DC_3a_6)
    
    

    
    all_Iin_DC_3a_6 = scaler_Iin.transform(df_IinB_DC_3a_6)
    Iin_train_arr_DC_3a_6 = scaler_Iin.transform(df_train_Iin_DC_3a_6)
    Iin_val_arr_DC_3a_6 = scaler_Iin.transform(df_val_Iin_DC_3a_6)
    Iin_test_arr_DC_3a_6 = scaler_Iin.transform(df_test_Iin_DC_3a_6)
    
    
    

    
    all_Vout_DC_3a_6 = scaler_Vout.transform(df_VoutB_DC_3a_6)
    Vout_train_arr_DC_3a_6 = scaler_Vout.transform(df_train_Vout_DC_3a_6)
    Vout_val_arr_DC_3a_6 = scaler_Vout.transform(df_val_Vout_DC_3a_6)
    Vout_test_arr_DC_3a_6 = scaler_Vout.transform(df_test_Vout_DC_3a_6)
    
    

    
    all_Iout_DC_3a_6 = scaler_Iout.transform(df_IoutB_DC_3a_6)
    Iout_train_arr_DC_3a_6 = scaler_Iout.transform(df_train_Iout_DC_3a_6)
    Iout_val_arr_DC_3a_6 = scaler_Iout.transform(df_val_Iout_DC_3a_6)
    Iout_test_arr_DC_3a_6 = scaler_Iout.transform(df_test_Iout_DC_3a_6)
    
    
    
    
    
    
    
    
    
    
    Vin_train_DC_3a_6 = transform_data(Vin_train_arr_DC_3a_6, seq_len)
    Vout_train_DC_3a_6 = transform_data(Vout_train_arr_DC_3a_6, seq_len)
    
    Vin_val_DC_3a_6 = transform_data(Vin_val_arr_DC_3a_6, seq_len)
    Vout_val_DC_3a_6 = transform_data(Vout_val_arr_DC_3a_6, seq_len)
    
    Vin_test_DC_3a_6 = transform_data(Vin_test_arr_DC_3a_6, seq_len)
    Vout_test_DC_3a_6 = transform_data(Vout_test_arr_DC_3a_6, seq_len)
    
    
    Vin_all_DC_3a_6 = transform_data(all_Vin_DC_3a_6, seq_len)
    Vout_all_DC_3a_6 = transform_data(all_Vout_DC_3a_6, seq_len)
    
    
    
    
    Iin_train_DC_3a_6 = transform_data(Iin_train_arr_DC_3a_6, seq_len)
    Iout_train_DC_3a_6 = transform_data(Iout_train_arr_DC_3a_6, seq_len)
    
    Iin_val_DC_3a_6 = transform_data(Iin_val_arr_DC_3a_6, seq_len)
    Iout_val_DC_3a_6 = transform_data(Iout_val_arr_DC_3a_6, seq_len)
    
    Iin_test_DC_3a_6 = transform_data(Iin_test_arr_DC_3a_6, seq_len)
    Iout_test_DC_3a_6 = transform_data(Iout_test_arr_DC_3a_6, seq_len)
    
    
    Iin_all_DC_3a_6 = transform_data(all_Iin_DC_3a_6, seq_len)
    Iout_all_DC_3a_6 = transform_data(all_Iout_DC_3a_6, seq_len)
    
    
    
    
    
    
    
    
    
    
    

    
    all_Vin_Pulse_5b_9 = scaler_Vin.transform(df_VinB_Pulse_5b_9)
    Vin_train_arr_Pulse_5b_9 = scaler_Vin.transform(df_train_Vin_Pulse_5b_9)
    Vin_val_arr_Pulse_5b_9 = scaler_Vin.transform(df_val_Vin_Pulse_5b_9)
    Vin_test_arr_Pulse_5b_9 = scaler_Vin.transform(df_test_Vin_Pulse_5b_9)
    
    

    
    all_Iin_Pulse_5b_9 = scaler_Iin.transform(df_IinB_Pulse_5b_9)
    Iin_train_arr_Pulse_5b_9 = scaler_Iin.transform(df_train_Iin_Pulse_5b_9)
    Iin_val_arr_Pulse_5b_9 = scaler_Iin.transform(df_val_Iin_Pulse_5b_9)
    Iin_test_arr_Pulse_5b_9 = scaler_Iin.transform(df_test_Iin_Pulse_5b_9)
    
    
    

    
    all_Vout_Pulse_5b_9 = scaler_Vout.transform(df_VoutB_Pulse_5b_9)
    Vout_train_arr_Pulse_5b_9 = scaler_Vout.transform(df_train_Vout_Pulse_5b_9)
    Vout_val_arr_Pulse_5b_9 = scaler_Vout.transform(df_val_Vout_Pulse_5b_9)
    Vout_test_arr_Pulse_5b_9 = scaler_Vout.transform(df_test_Vout_Pulse_5b_9)
    
    

    
    all_Iout_Pulse_5b_9 = scaler_Iout.transform(df_IoutB_Pulse_5b_9)
    Iout_train_arr_Pulse_5b_9 = scaler_Iout.transform(df_train_Iout_Pulse_5b_9)
    Iout_val_arr_Pulse_5b_9 = scaler_Iout.transform(df_val_Iout_Pulse_5b_9)
    Iout_test_arr_Pulse_5b_9 = scaler_Iout.transform(df_test_Iout_Pulse_5b_9)
    
    
    
    
    
    
    
    
    
    
    Vin_train_Pulse_5b_9 = transform_data(Vin_train_arr_Pulse_5b_9, seq_len)
    Vout_train_Pulse_5b_9 = transform_data(Vout_train_arr_Pulse_5b_9, seq_len)
    
    Vin_val_Pulse_5b_9 = transform_data(Vin_val_arr_Pulse_5b_9, seq_len)
    Vout_val_Pulse_5b_9 = transform_data(Vout_val_arr_Pulse_5b_9, seq_len)
    
    Vin_test_Pulse_5b_9 = transform_data(Vin_test_arr_Pulse_5b_9, seq_len)
    Vout_test_Pulse_5b_9 = transform_data(Vout_test_arr_Pulse_5b_9, seq_len)
    
    
    Vin_all_Pulse_5b_9 = transform_data(all_Vin_Pulse_5b_9, seq_len)
    Vout_all_Pulse_5b_9 = transform_data(all_Vout_Pulse_5b_9, seq_len)
    
    
    
    
    Iin_train_Pulse_5b_9 = transform_data(Iin_train_arr_Pulse_5b_9, seq_len)
    Iout_train_Pulse_5b_9 = transform_data(Iout_train_arr_Pulse_5b_9, seq_len)
    
    Iin_val_Pulse_5b_9 = transform_data(Iin_val_arr_Pulse_5b_9, seq_len)
    Iout_val_Pulse_5b_9 = transform_data(Iout_val_arr_Pulse_5b_9, seq_len)
    
    Iin_test_Pulse_5b_9 = transform_data(Iin_test_arr_Pulse_5b_9, seq_len)
    Iout_test_Pulse_5b_9 = transform_data(Iout_test_arr_Pulse_5b_9, seq_len)
    
    
    Iin_all_Pulse_5b_9 = transform_data(all_Iin_Pulse_5b_9, seq_len)
    Iout_all_Pulse_5b_9 = transform_data(all_Iout_Pulse_5b_9, seq_len)
    
    
    
    
    
    
    
    
    
    

    
    all_Vin_Pulse_6c_5 = scaler_Vin.transform(df_VinB_Pulse_6c_5)
    Vin_train_arr_Pulse_6c_5 = scaler_Vin.transform(df_train_Vin_Pulse_6c_5)
    Vin_val_arr_Pulse_6c_5 = scaler_Vin.transform(df_val_Vin_Pulse_6c_5)
    Vin_test_arr_Pulse_6c_5 = scaler_Vin.transform(df_test_Vin_Pulse_6c_5)
    
    

    
    all_Iin_Pulse_6c_5 = scaler_Iin.transform(df_IinB_Pulse_6c_5)
    Iin_train_arr_Pulse_6c_5 = scaler_Iin.transform(df_train_Iin_Pulse_6c_5)
    Iin_val_arr_Pulse_6c_5 = scaler_Iin.transform(df_val_Iin_Pulse_6c_5)
    Iin_test_arr_Pulse_6c_5 = scaler_Iin.transform(df_test_Iin_Pulse_6c_5)
    
    
    

    
    all_Vout_Pulse_6c_5 = scaler_Vout.transform(df_VoutB_Pulse_6c_5)
    Vout_train_arr_Pulse_6c_5 = scaler_Vout.transform(df_train_Vout_Pulse_6c_5)
    Vout_val_arr_Pulse_6c_5 = scaler_Vout.transform(df_val_Vout_Pulse_6c_5)
    Vout_test_arr_Pulse_6c_5 = scaler_Vout.transform(df_test_Vout_Pulse_6c_5)
    
    

    
    all_Iout_Pulse_6c_5 = scaler_Iout.transform(df_IoutB_Pulse_6c_5)
    Iout_train_arr_Pulse_6c_5 = scaler_Iout.transform(df_train_Iout_Pulse_6c_5)
    Iout_val_arr_Pulse_6c_5 = scaler_Iout.transform(df_val_Iout_Pulse_6c_5)
    Iout_test_arr_Pulse_6c_5 = scaler_Iout.transform(df_test_Iout_Pulse_6c_5)
    
    
    
    
    
    
    
    
    
    
    Vin_train_Pulse_6c_5 = transform_data(Vin_train_arr_Pulse_6c_5, seq_len)
    Vout_train_Pulse_6c_5 = transform_data(Vout_train_arr_Pulse_6c_5, seq_len)
    
    Vin_val_Pulse_6c_5 = transform_data(Vin_val_arr_Pulse_6c_5, seq_len)
    Vout_val_Pulse_6c_5 = transform_data(Vout_val_arr_Pulse_6c_5, seq_len)
    
    Vin_test_Pulse_6c_5 = transform_data(Vin_test_arr_Pulse_6c_5, seq_len)
    Vout_test_Pulse_6c_5 = transform_data(Vout_test_arr_Pulse_6c_5, seq_len)
    
    
    Vin_all_Pulse_6c_5 = transform_data(all_Vin_Pulse_6c_5, seq_len)
    Vout_all_Pulse_6c_5 = transform_data(all_Vout_Pulse_6c_5, seq_len)
    
    
    
    
    Iin_train_Pulse_6c_5 = transform_data(Iin_train_arr_Pulse_6c_5, seq_len)
    Iout_train_Pulse_6c_5 = transform_data(Iout_train_arr_Pulse_6c_5, seq_len)
    
    Iin_val_Pulse_6c_5 = transform_data(Iin_val_arr_Pulse_6c_5, seq_len)
    Iout_val_Pulse_6c_5 = transform_data(Iout_val_arr_Pulse_6c_5, seq_len)
    
    Iin_test_Pulse_6c_5 = transform_data(Iin_test_arr_Pulse_6c_5, seq_len)
    Iout_test_Pulse_6c_5 = transform_data(Iout_test_arr_Pulse_6c_5, seq_len)
    
    
    Iin_all_Pulse_6c_5 = transform_data(all_Iin_Pulse_6c_5, seq_len)
    Iout_all_Pulse_6c_5 = transform_data(all_Iout_Pulse_6c_5, seq_len)
    
    
    
    
    
    
    
    
    

    
    all_Vin_PWM3 = scaler_Vin.transform(df_VinB_PWM3)
    Vin_train_arr_PWM3 = scaler_Vin.transform(df_train_Vin_PWM3)
    Vin_val_arr_PWM3 = scaler_Vin.transform(df_val_Vin_PWM3)
    Vin_test_arr_PWM3 = scaler_Vin.transform(df_test_Vin_PWM3)
    
    

    
    all_Iin_PWM3 = scaler_Iin.transform(df_IinB_PWM3)
    Iin_train_arr_PWM3 = scaler_Iin.transform(df_train_Iin_PWM3)
    Iin_val_arr_PWM3 = scaler_Iin.transform(df_val_Iin_PWM3)
    Iin_test_arr_PWM3 = scaler_Iin.transform(df_test_Iin_PWM3)
    
    
    

    
    all_Vout_PWM3 = scaler_Vout.transform(df_VoutB_PWM3)
    Vout_train_arr_PWM3 = scaler_Vout.transform(df_train_Vout_PWM3)
    Vout_val_arr_PWM3 = scaler_Vout.transform(df_val_Vout_PWM3)
    Vout_test_arr_PWM3 = scaler_Vout.transform(df_test_Vout_PWM3)
    
    

    
    all_Iout_PWM3 = scaler_Iout.transform(df_IoutB_PWM3)
    Iout_train_arr_PWM3 = scaler_Iout.transform(df_train_Iout_PWM3)
    Iout_val_arr_PWM3 = scaler_Iout.transform(df_val_Iout_PWM3)
    Iout_test_arr_PWM3 = scaler_Iout.transform(df_test_Iout_PWM3)
    
    
    
    
    
    
    
    
    
    
    Vin_train_PWM3 = transform_data(Vin_train_arr_PWM3, seq_len)
    Vout_train_PWM3 = transform_data(Vout_train_arr_PWM3, seq_len)
    
    Vin_val_PWM3 = transform_data(Vin_val_arr_PWM3, seq_len)
    Vout_val_PWM3 = transform_data(Vout_val_arr_PWM3, seq_len)
    
    Vin_test_PWM3 = transform_data(Vin_test_arr_PWM3, seq_len)
    Vout_test_PWM3 = transform_data(Vout_test_arr_PWM3, seq_len)
    
    
    Vin_all_PWM3 = transform_data(all_Vin_PWM3, seq_len)
    Vout_all_PWM3 = transform_data(all_Vout_PWM3, seq_len)
    
    
    
    
    Iin_train_PWM3 = transform_data(Iin_train_arr_PWM3, seq_len)
    Iout_train_PWM3 = transform_data(Iout_train_arr_PWM3, seq_len)
    
    Iin_val_PWM3 = transform_data(Iin_val_arr_PWM3, seq_len)
    Iout_val_PWM3 = transform_data(Iout_val_arr_PWM3, seq_len)
    
    Iin_test_PWM3 = transform_data(Iin_test_arr_PWM3, seq_len)
    Iout_test_PWM3 = transform_data(Iout_test_arr_PWM3, seq_len)
    
    
    Iin_all_PWM3 = transform_data(all_Iin_PWM3, seq_len)
    Iout_all_PWM3 = transform_data(all_Iout_PWM3, seq_len)
    
    
    
    
    
    
    
    

    
    all_Vin_DC_1b_6 = scaler_Vin.transform(df_VinB_DC_1b_6)
    Vin_train_arr_DC_1b_6 = scaler_Vin.transform(df_train_Vin_DC_1b_6)
    Vin_val_arr_DC_1b_6 = scaler_Vin.transform(df_val_Vin_DC_1b_6)
    Vin_test_arr_DC_1b_6 = scaler_Vin.transform(df_test_Vin_DC_1b_6)
    
    

    
    all_Iin_DC_1b_6 = scaler_Iin.transform(df_IinB_DC_1b_6)
    Iin_train_arr_DC_1b_6 = scaler_Iin.transform(df_train_Iin_DC_1b_6)
    Iin_val_arr_DC_1b_6 = scaler_Iin.transform(df_val_Iin_DC_1b_6)
    Iin_test_arr_DC_1b_6 = scaler_Iin.transform(df_test_Iin_DC_1b_6)
    
    
    

    
    all_Vout_DC_1b_6 = scaler_Vout.transform(df_VoutB_DC_1b_6)
    Vout_train_arr_DC_1b_6 = scaler_Vout.transform(df_train_Vout_DC_1b_6)
    Vout_val_arr_DC_1b_6 = scaler_Vout.transform(df_val_Vout_DC_1b_6)
    Vout_test_arr_DC_1b_6 = scaler_Vout.transform(df_test_Vout_DC_1b_6)
    
    

    
    all_Iout_DC_1b_6 = scaler_Iout.transform(df_IoutB_DC_1b_6)
    Iout_train_arr_DC_1b_6 = scaler_Iout.transform(df_train_Iout_DC_1b_6)
    Iout_val_arr_DC_1b_6 = scaler_Iout.transform(df_val_Iout_DC_1b_6)
    Iout_test_arr_DC_1b_6 = scaler_Iout.transform(df_test_Iout_DC_1b_6)
    
    
    
    
    
    
    
    
    
    
    Vin_train_DC_1b_6 = transform_data(Vin_train_arr_DC_1b_6, seq_len)
    Vout_train_DC_1b_6 = transform_data(Vout_train_arr_DC_1b_6, seq_len)
    
    Vin_val_DC_1b_6 = transform_data(Vin_val_arr_DC_1b_6, seq_len)
    Vout_val_DC_1b_6 = transform_data(Vout_val_arr_DC_1b_6, seq_len)
    
    Vin_test_DC_1b_6 = transform_data(Vin_test_arr_DC_1b_6, seq_len)
    Vout_test_DC_1b_6 = transform_data(Vout_test_arr_DC_1b_6, seq_len)
    
    
    Vin_all_DC_1b_6 = transform_data(all_Vin_DC_1b_6, seq_len)
    Vout_all_DC_1b_6 = transform_data(all_Vout_DC_1b_6, seq_len)
    
    
    
    
    Iin_train_DC_1b_6 = transform_data(Iin_train_arr_DC_1b_6, seq_len)
    Iout_train_DC_1b_6 = transform_data(Iout_train_arr_DC_1b_6, seq_len)
    
    Iin_val_DC_1b_6 = transform_data(Iin_val_arr_DC_1b_6, seq_len)
    Iout_val_DC_1b_6 = transform_data(Iout_val_arr_DC_1b_6, seq_len)
    
    Iin_test_DC_1b_6 = transform_data(Iin_test_arr_DC_1b_6, seq_len)
    Iout_test_DC_1b_6 = transform_data(Iout_test_arr_DC_1b_6, seq_len)
    
    
    Iin_all_DC_1b_6 = transform_data(all_Iin_DC_1b_6, seq_len)
    Iout_all_DC_1b_6 = transform_data(all_Iout_DC_1b_6, seq_len)
    
    
    
    
    
    Vin_train_DC_2b_6=Vin_train_DC_2b_6.to(device)
    Iout_train_DC_2b_6=Iout_train_DC_2b_6.to(device) 
    Vout_train_DC_2b_6=Vout_train_DC_2b_6.to(device) 
    Iin_train_DC_2b_6=Iin_train_DC_2b_6.to(device)

    Vin_val_DC_2b_6=Vin_val_DC_2b_6.to(device)
    Iin_val_DC_2b_6=Iin_val_DC_2b_6.to(device)
    Vout_val_DC_2b_6=Vout_val_DC_2b_6.to(device)
    Iout_val_DC_2b_6=Iout_val_DC_2b_6.to(device)




    Vin_train_DC_2c_8=Vin_train_DC_2c_8.to(device)
    Iout_train_DC_2c_8=Iout_train_DC_2c_8.to(device) 
    Vout_train_DC_2c_8=Vout_train_DC_2c_8.to(device) 
    Iin_train_DC_2c_8=Iin_train_DC_2c_8.to(device)

    Vin_val_DC_2c_8=Vin_val_DC_2c_8.to(device)
    Iin_val_DC_2c_8=Iin_val_DC_2c_8.to(device)
    Vout_val_DC_2c_8=Vout_val_DC_2c_8.to(device)
    Iout_val_DC_2c_8=Iout_val_DC_2c_8.to(device)





    Vin_train_DC_3a_6=Vin_train_DC_3a_6.to(device)
    Iout_train_DC_3a_6=Iout_train_DC_3a_6.to(device) 
    Vout_train_DC_3a_6=Vout_train_DC_3a_6.to(device) 
    Iin_train_DC_3a_6=Iin_train_DC_3a_6.to(device)

    Vin_val_DC_3a_6=Vin_val_DC_3a_6.to(device)
    Iin_val_DC_3a_6=Iin_val_DC_3a_6.to(device)
    Vout_val_DC_3a_6=Vout_val_DC_3a_6.to(device)
    Iout_val_DC_3a_6=Iout_val_DC_3a_6.to(device)






    Vin_train_Pulse_5b_9=Vin_train_Pulse_5b_9.to(device)
    Iout_train_Pulse_5b_9=Iout_train_Pulse_5b_9.to(device) 
    Vout_train_Pulse_5b_9=Vout_train_Pulse_5b_9.to(device) 
    Iin_train_Pulse_5b_9=Iin_train_Pulse_5b_9.to(device)

    Vin_val_Pulse_5b_9=Vin_val_Pulse_5b_9.to(device)
    Iin_val_Pulse_5b_9=Iin_val_Pulse_5b_9.to(device)
    Vout_val_Pulse_5b_9=Vout_val_Pulse_5b_9.to(device)
    Iout_val_Pulse_5b_9=Iout_val_Pulse_5b_9.to(device)






    Vin_train_Pulse_6c_5=Vin_train_Pulse_6c_5.to(device)
    Iout_train_Pulse_6c_5=Iout_train_Pulse_6c_5.to(device) 
    Vout_train_Pulse_6c_5=Vout_train_Pulse_6c_5.to(device) 
    Iin_train_Pulse_6c_5=Iin_train_Pulse_6c_5.to(device)

    Vin_val_Pulse_6c_5=Vin_val_Pulse_6c_5.to(device)
    Iin_val_Pulse_6c_5=Iin_val_Pulse_6c_5.to(device)
    Vout_val_Pulse_6c_5=Vout_val_Pulse_6c_5.to(device)
    Iout_val_Pulse_6c_5=Iout_val_Pulse_6c_5.to(device)





    Vin_train_PWM3=Vin_train_PWM3.to(device)
    Iout_train_PWM3=Iout_train_PWM3.to(device) 
    Vout_train_PWM3=Vout_train_PWM3.to(device) 
    Iin_train_PWM3=Iin_train_PWM3.to(device)

    Vin_val_PWM3=Vin_val_PWM3.to(device)
    Iin_val_PWM3=Iin_val_PWM3.to(device)
    Vout_val_PWM3=Vout_val_PWM3.to(device)
    Iout_val_PWM3=Iout_val_PWM3.to(device)






    Vin_train_DC_1b_6=Vin_train_DC_1b_6.to(device)
    Iout_train_DC_1b_6=Iout_train_DC_1b_6.to(device) 
    Vout_train_DC_1b_6=Vout_train_DC_1b_6.to(device) 
    Iin_train_DC_1b_6=Iin_train_DC_1b_6.to(device)

    Vin_val_DC_1b_6=Vin_val_DC_1b_6.to(device)
    Iin_val_DC_1b_6=Iin_val_DC_1b_6.to(device)
    Vout_val_DC_1b_6=Vout_val_DC_1b_6.to(device)
    Iout_val_DC_1b_6=Iout_val_DC_1b_6.to(device)





    Vin_all_DC_1b_6=Vin_all_DC_1b_6.to(device)
    Iout_all_DC_1b_6=Iout_all_DC_1b_6.to(device)

    Vin_all_SineSweep_10_6=Vin_all_SineSweep_10_6.to(device)
    Iout_all_SineSweep_10_6=Iout_all_SineSweep_10_6.to(device)






    Vin_train_SineSweep_10_6=Vin_train_SineSweep_10_6.to(device)
    Iout_train_SineSweep_10_6=Iout_train_SineSweep_10_6.to(device) 
    Vout_train_SineSweep_10_6=Vout_train_SineSweep_10_6.to(device) 
    Iin_train_SineSweep_10_6=Iin_train_SineSweep_10_6.to(device)

    Vin_val_SineSweep_10_6=Vin_val_SineSweep_10_6.to(device)
    Iin_val_SineSweep_10_6=Iin_val_SineSweep_10_6.to(device)
    Vout_val_SineSweep_10_6=Vout_val_SineSweep_10_6.to(device)
    Iout_val_SineSweep_10_6=Iout_val_SineSweep_10_6.to(device)




    return scaler_Vin, scaler_Iin, scaler_Vout, scaler_Iout, Vin_train_SineSweep_10_6, Vout_train_SineSweep_10_6, Vin_val_SineSweep_10_6, Vout_val_SineSweep_10_6, Vin_test_SineSweep_10_6, Vout_test_SineSweep_10_6, Vin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iin_train_SineSweep_10_6, Iout_train_SineSweep_10_6, Iin_val_SineSweep_10_6, Iout_val_SineSweep_10_6, Iin_test_SineSweep_10_6, Iout_test_SineSweep_10_6, Iin_all_SineSweep_10_6, Iout_all_SineSweep_10_6 , Vin_train_DC_2b_6, Vout_train_DC_2b_6, Vin_val_DC_2b_6, Vout_val_DC_2b_6, Vin_test_DC_2b_6, Vout_test_DC_2b_6, Vin_all_DC_2b_6, Vout_all_DC_2b_6, Iin_train_DC_2b_6, Iout_train_DC_2b_6, Iin_val_DC_2b_6, Iout_val_DC_2b_6, Iin_test_DC_2b_6, Iout_test_DC_2b_6, Iin_all_DC_2b_6, Iout_all_DC_2b_6, Vin_train_DC_2c_8, Vout_train_DC_2c_8, Vin_val_DC_2c_8, Vout_val_DC_2c_8, Vin_test_DC_2c_8, Vout_test_DC_2c_8, Vin_all_DC_2c_8, Vout_all_DC_2c_8, Iin_train_DC_2c_8, Iout_train_DC_2c_8, Iin_val_DC_2c_8, Iout_val_DC_2c_8, Iin_test_DC_2c_8, Iout_test_DC_2c_8, Iin_all_DC_2c_8, Iout_all_DC_2c_8, Vin_train_DC_3a_6, Vout_train_DC_3a_6, Vin_val_DC_3a_6, Vout_val_DC_3a_6, Vin_test_DC_3a_6, Vout_test_DC_3a_6, Vin_all_DC_3a_6, Vout_all_DC_3a_6, Iin_train_DC_3a_6, Iout_train_DC_3a_6, Iin_val_DC_3a_6, Iout_val_DC_3a_6, Iin_test_DC_3a_6, Iout_test_DC_3a_6, Iin_all_DC_3a_6, Iout_all_DC_3a_6, Vin_train_Pulse_5b_9, Vout_train_Pulse_5b_9, Vin_val_Pulse_5b_9, Vout_val_Pulse_5b_9, Vin_test_Pulse_5b_9, Vout_test_Pulse_5b_9, Vin_all_Pulse_5b_9, Vout_all_Pulse_5b_9, Iin_train_Pulse_5b_9, Iout_train_Pulse_5b_9, Iin_val_Pulse_5b_9, Iout_val_Pulse_5b_9, Iin_test_Pulse_5b_9, Iout_test_Pulse_5b_9, Iin_all_Pulse_5b_9, Iout_all_Pulse_5b_9, Vin_train_Pulse_6c_5, Vout_train_Pulse_6c_5, Vin_val_Pulse_6c_5, Vout_val_Pulse_6c_5, Vin_test_Pulse_6c_5, Vout_test_Pulse_6c_5, Vin_all_Pulse_6c_5, Vout_all_Pulse_6c_5, Iin_train_Pulse_6c_5, Iout_train_Pulse_6c_5, Iin_val_Pulse_6c_5, Iout_val_Pulse_6c_5, Iin_test_Pulse_6c_5, Iout_test_Pulse_6c_5, Iin_all_Pulse_6c_5, Iout_all_Pulse_6c_5, Vin_train_PWM3, Vout_train_PWM3, Vin_val_PWM3, Vout_val_PWM3, Vin_test_PWM3, Vout_test_PWM3, Vin_all_PWM3, Vout_all_PWM3, Iin_train_PWM3, Iout_train_PWM3, Iin_val_PWM3, Iout_val_PWM3, Iin_test_PWM3, Iout_test_PWM3, Iin_all_PWM3, Iout_all_PWM3, Vin_train_DC_1b_6, Vout_train_DC_1b_6, Vin_val_DC_1b_6, Vout_val_DC_1b_6, Vin_test_DC_1b_6, Vout_test_DC_1b_6, Vin_all_DC_1b_6, Vout_all_DC_1b_6, Iin_train_DC_1b_6, Iout_train_DC_1b_6, Iin_val_DC_1b_6, Iout_val_DC_1b_6, Iin_test_DC_1b_6, Iout_test_DC_1b_6, Iin_all_DC_1b_6, Iout_all_DC_1b_6, df_Vin_SineSweep_10_6, df_Iin_SineSweep_10_6, df_Vout_SineSweep_10_6, df_Iout_SineSweep_10_6





















