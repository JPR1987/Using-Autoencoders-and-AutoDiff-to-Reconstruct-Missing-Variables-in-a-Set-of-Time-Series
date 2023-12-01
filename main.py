# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:02:42 2023

@author: hidden
"""

import glob
from platform import python_version
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import time
import torch.nn as nn
import torch.optim as optim


#cuda
print("GPU avaiable:", torch.cuda.is_available())

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev) 


print("python version==%s" % python_version())
print("pandas==%s" % pd.__version__)
print("numpy==%s" % np.__version__)
print("sklearn==%s" % sklearn.__version__)
print("torch==%s" % torch.__version__)
print("matplotlib==%s" % matplotlib.__version__)

plt.rcParams[
    "figure.facecolor"
] = "w"  # force white background on plots when using dark mode in JupyterLab










#data_import
#load_and_transform_all_data()

seq_len = 3  # <->"How many time steps are considered."

from data_import import load_and_transform_all_data

(scaler_Vin, 
scaler_Iin, 
scaler_Vout, 
scaler_Iout, 
Vin_train_SineSweep_10_6, 
Vout_train_SineSweep_10_6, 
Vin_val_SineSweep_10_6, 
Vout_val_SineSweep_10_6, 
Vin_test_SineSweep_10_6, 
Vout_test_SineSweep_10_6, 
Vin_all_SineSweep_10_6, 
Vout_all_SineSweep_10_6, 
Iin_train_SineSweep_10_6, 
Iout_train_SineSweep_10_6, 
Iin_val_SineSweep_10_6, 
Iout_val_SineSweep_10_6, 
Iin_test_SineSweep_10_6, 
Iout_test_SineSweep_10_6, 
Iin_all_SineSweep_10_6, 
Iout_all_SineSweep_10_6, 
Vin_train_DC_2b_6, 
Vout_train_DC_2b_6, 
Vin_val_DC_2b_6, 
Vout_val_DC_2b_6, 
Vin_test_DC_2b_6, 
Vout_test_DC_2b_6, 
Vin_all_DC_2b_6, 
Vout_all_DC_2b_6, 
Iin_train_DC_2b_6, 
Iout_train_DC_2b_6, 
Iin_val_DC_2b_6, 
Iout_val_DC_2b_6, 
Iin_test_DC_2b_6, 
Iout_test_DC_2b_6, 
Iin_all_DC_2b_6, 
Iout_all_DC_2b_6, 
Vin_train_DC_2c_8, 
Vout_train_DC_2c_8, 
Vin_val_DC_2c_8, 
Vout_val_DC_2c_8, 
Vin_test_DC_2c_8, 
Vout_test_DC_2c_8, 
Vin_all_DC_2c_8, 
Vout_all_DC_2c_8, 
Iin_train_DC_2c_8, 
Iout_train_DC_2c_8, 
Iin_val_DC_2c_8, 
Iout_val_DC_2c_8, 
Iin_test_DC_2c_8, 
Iout_test_DC_2c_8, 
Iin_all_DC_2c_8, 
Iout_all_DC_2c_8, 
Vin_train_DC_3a_6, 
Vout_train_DC_3a_6, 
Vin_val_DC_3a_6, 
Vout_val_DC_3a_6, 
Vin_test_DC_3a_6, 
Vout_test_DC_3a_6, 
Vin_all_DC_3a_6, 
Vout_all_DC_3a_6, 
Iin_train_DC_3a_6, 
Iout_train_DC_3a_6, 
Iin_val_DC_3a_6, 
Iout_val_DC_3a_6, 
Iin_test_DC_3a_6, 
Iout_test_DC_3a_6, 
Iin_all_DC_3a_6, 
Iout_all_DC_3a_6, 
Vin_train_Pulse_5b_9, 
Vout_train_Pulse_5b_9, 
Vin_val_Pulse_5b_9, 
Vout_val_Pulse_5b_9, 
Vin_test_Pulse_5b_9, 
Vout_test_Pulse_5b_9, 
Vin_all_Pulse_5b_9, 
Vout_all_Pulse_5b_9, 
Iin_train_Pulse_5b_9, 
Iout_train_Pulse_5b_9, 
Iin_val_Pulse_5b_9, 
Iout_val_Pulse_5b_9, 
Iin_test_Pulse_5b_9, 
Iout_test_Pulse_5b_9, 
Iin_all_Pulse_5b_9, 
Iout_all_Pulse_5b_9, 
Vin_train_Pulse_6c_5, 
Vout_train_Pulse_6c_5, 
Vin_val_Pulse_6c_5, 
Vout_val_Pulse_6c_5, 
Vin_test_Pulse_6c_5, 
Vout_test_Pulse_6c_5, 
Vin_all_Pulse_6c_5, 
Vout_all_Pulse_6c_5, 
Iin_train_Pulse_6c_5, 
Iout_train_Pulse_6c_5, 
Iin_val_Pulse_6c_5, 
Iout_val_Pulse_6c_5, 
Iin_test_Pulse_6c_5, 
Iout_test_Pulse_6c_5, 
Iin_all_Pulse_6c_5, 
Iout_all_Pulse_6c_5, 
Vin_train_PWM3, 
Vout_train_PWM3, 
Vin_val_PWM3, 
Vout_val_PWM3, 
Vin_test_PWM3, 
Vout_test_PWM3, 
Vin_all_PWM3, 
Vout_all_PWM3, 
Iin_train_PWM3, 
Iout_train_PWM3, 
Iin_val_PWM3, 
Iout_val_PWM3, 
Iin_test_PWM3, 
Iout_test_PWM3, 
Iin_all_PWM3, 
Iout_all_PWM3, 
Vin_train_DC_1b_6, 
Vout_train_DC_1b_6, 
Vin_val_DC_1b_6, 
Vout_val_DC_1b_6, 
Vin_test_DC_1b_6, 
Vout_test_DC_1b_6, 
Vin_all_DC_1b_6, 
Vout_all_DC_1b_6, 
Iin_train_DC_1b_6, 
Iout_train_DC_1b_6, 
Iin_val_DC_1b_6, 
Iout_val_DC_1b_6, 
Iin_test_DC_1b_6, 
Iout_test_DC_1b_6, 
Iin_all_DC_1b_6, 
Iout_all_DC_1b_6, 
df_Vin_SineSweep_10_6, 
df_Iin_SineSweep_10_6, 
df_Vout_SineSweep_10_6, 
df_Iout_SineSweep_10_6) = load_and_transform_all_data(seq_len, device)









#define network
inputsize = 4
hiddensize1 = 40
hiddensize2 = 15
hiddensize3 = 4
hiddensize4 = 15
hiddensize5 = 40
#outputsize = 4


from net import network_model

model = network_model(inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)
model.to(device)










#training model and save
runs=1000
epoches=1 # best results if epoches=1; training more shuffled, trained model less dependend on last training dataset

lossfn = nn.MSELoss(reduction='sum')
lossfn = lossfn.to(device)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

save_path_training = 'model/trained_model_and_optimizer_checkpoint.pth'


from training import run_training
model, optimizer, epoch_last, run_last = run_training(runs, epoches, lossfn, optimizer, model, 
                                                      Vin_train_SineSweep_10_6, 
                                                      Vout_train_SineSweep_10_6, 
                                                      Vin_val_SineSweep_10_6, 
                                                      Vout_val_SineSweep_10_6, 
                                                      Vin_test_SineSweep_10_6, 
                                                      Vout_test_SineSweep_10_6, 
                                                      Vin_all_SineSweep_10_6, 
                                                      Vout_all_SineSweep_10_6, 
                                                      Iin_train_SineSweep_10_6, 
                                                      Iout_train_SineSweep_10_6, 
                                                      Iin_val_SineSweep_10_6, 
                                                      Iout_val_SineSweep_10_6, 
                                                      Iin_test_SineSweep_10_6, 
                                                      Iout_test_SineSweep_10_6, 
                                                      Iin_all_SineSweep_10_6, 
                                                      Iout_all_SineSweep_10_6, 
                                                      Vin_train_DC_2b_6, 
                                                      Vout_train_DC_2b_6, 
                                                      Vin_val_DC_2b_6, 
                                                      Vout_val_DC_2b_6, 
                                                      Vin_test_DC_2b_6, 
                                                      Vout_test_DC_2b_6, 
                                                      Vin_all_DC_2b_6, 
                                                      Vout_all_DC_2b_6, 
                                                      Iin_train_DC_2b_6, 
                                                      Iout_train_DC_2b_6, 
                                                      Iin_val_DC_2b_6, 
                                                      Iout_val_DC_2b_6, 
                                                      Iin_test_DC_2b_6, 
                                                      Iout_test_DC_2b_6, 
                                                      Iin_all_DC_2b_6, 
                                                      Iout_all_DC_2b_6, 
                                                      Vin_train_DC_2c_8, 
                                                      Vout_train_DC_2c_8, 
                                                      Vin_val_DC_2c_8, 
                                                      Vout_val_DC_2c_8, 
                                                      Vin_test_DC_2c_8, 
                                                      Vout_test_DC_2c_8, 
                                                      Vin_all_DC_2c_8, 
                                                      Vout_all_DC_2c_8, 
                                                      Iin_train_DC_2c_8, 
                                                      Iout_train_DC_2c_8, 
                                                      Iin_val_DC_2c_8, 
                                                      Iout_val_DC_2c_8, 
                                                      Iin_test_DC_2c_8, 
                                                      Iout_test_DC_2c_8, 
                                                      Iin_all_DC_2c_8, 
                                                      Iout_all_DC_2c_8, 
                                                      Vin_train_DC_3a_6, 
                                                      Vout_train_DC_3a_6, 
                                                      Vin_val_DC_3a_6, 
                                                      Vout_val_DC_3a_6, 
                                                      Vin_test_DC_3a_6, 
                                                      Vout_test_DC_3a_6, 
                                                      Vin_all_DC_3a_6, 
                                                      Vout_all_DC_3a_6, 
                                                      Iin_train_DC_3a_6, 
                                                      Iout_train_DC_3a_6, 
                                                      Iin_val_DC_3a_6, 
                                                      Iout_val_DC_3a_6, 
                                                      Iin_test_DC_3a_6, 
                                                      Iout_test_DC_3a_6, 
                                                      Iin_all_DC_3a_6, 
                                                      Iout_all_DC_3a_6, 
                                                      Vin_train_Pulse_5b_9, 
                                                      Vout_train_Pulse_5b_9, 
                                                      Vin_val_Pulse_5b_9, 
                                                      Vout_val_Pulse_5b_9, 
                                                      Vin_test_Pulse_5b_9, 
                                                      Vout_test_Pulse_5b_9, 
                                                      Vin_all_Pulse_5b_9, 
                                                      Vout_all_Pulse_5b_9, 
                                                      Iin_train_Pulse_5b_9, 
                                                      Iout_train_Pulse_5b_9, 
                                                      Iin_val_Pulse_5b_9, 
                                                      Iout_val_Pulse_5b_9, 
                                                      Iin_test_Pulse_5b_9, 
                                                      Iout_test_Pulse_5b_9, 
                                                      Iin_all_Pulse_5b_9, 
                                                      Iout_all_Pulse_5b_9, 
                                                      Vin_train_Pulse_6c_5, 
                                                      Vout_train_Pulse_6c_5, 
                                                      Vin_val_Pulse_6c_5, 
                                                      Vout_val_Pulse_6c_5, 
                                                      Vin_test_Pulse_6c_5, 
                                                      Vout_test_Pulse_6c_5, 
                                                      Vin_all_Pulse_6c_5, 
                                                      Vout_all_Pulse_6c_5, 
                                                      Iin_train_Pulse_6c_5, 
                                                      Iout_train_Pulse_6c_5, 
                                                      Iin_val_Pulse_6c_5, 
                                                      Iout_val_Pulse_6c_5, 
                                                      Iin_test_Pulse_6c_5, 
                                                      Iout_test_Pulse_6c_5, 
                                                      Iin_all_Pulse_6c_5, 
                                                      Iout_all_Pulse_6c_5, 
                                                      Vin_train_PWM3, 
                                                      Vout_train_PWM3, 
                                                      Vin_val_PWM3, 
                                                      Vout_val_PWM3, 
                                                      Vin_test_PWM3, 
                                                      Vout_test_PWM3, 
                                                      Vin_all_PWM3, 
                                                      Vout_all_PWM3, 
                                                      Iin_train_PWM3, 
                                                      Iout_train_PWM3, 
                                                      Iin_val_PWM3, 
                                                      Iout_val_PWM3, 
                                                      Iin_test_PWM3, 
                                                      Iout_test_PWM3, 
                                                      Iin_all_PWM3, 
                                                      Iout_all_PWM3, 
                                                      Vin_train_DC_1b_6, 
                                                      Vout_train_DC_1b_6, 
                                                      Vin_val_DC_1b_6, 
                                                      Vout_val_DC_1b_6, 
                                                      Vin_test_DC_1b_6, 
                                                      Vout_test_DC_1b_6, 
                                                      Vin_all_DC_1b_6, 
                                                      Vout_all_DC_1b_6, 
                                                      Iin_train_DC_1b_6, 
                                                      Iout_train_DC_1b_6, 
                                                      Iin_val_DC_1b_6, 
                                                      Iout_val_DC_1b_6, 
                                                      Iin_test_DC_1b_6, 
                                                      Iout_test_DC_1b_6, 
                                                      Iin_all_DC_1b_6, 
                                                      Iout_all_DC_1b_6)



from training import save_checkpoint_training
save_checkpoint_training(model, optimizer, save_path_training, epoch_last, run_last)





#Test with unknown dataset

from gen_seq_cpu import gen_seq
Vin_all_SineSweep_10_6_model, Iin_all_SineSweep_10_6_model, Vout_all_SineSweep_10_6_model, Iout_all_SineSweep_10_6_model = gen_seq(save_path_training, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)





#plot results (original vs. trained model)
plt.plot(df_Vin_SineSweep_10_6, label="Vin_original")
plt.plot(Vin_all_SineSweep_10_6_model, label="Vin_trained_model")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Voltage Vin / V")
plt.show()
plt.plot(df_Iin_SineSweep_10_6, label="Iin_original")
plt.plot(Iin_all_SineSweep_10_6_model, label="Iin_trained_model")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Current Iin / A")
plt.show()
plt.plot(df_Vout_SineSweep_10_6, label="Vout_original")
plt.plot(Vout_all_SineSweep_10_6_model, label="Vout_trained_model")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Voltage Vout / V")
plt.show()
plt.plot(df_Iout_SineSweep_10_6, label="Iout_original")
plt.plot(Iout_all_SineSweep_10_6_model, label="Iout_trained_model")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Current Iout / A")
plt.show()




#save_test_results_of_trained_model
np.savetxt('csv/Vin_all_SineSweep_10_6_test_model.txt', Vin_all_SineSweep_10_6_model, delimiter=',')
np.savetxt('csv/Iin_all_SineSweep_10_6_test_model.txt', Iin_all_SineSweep_10_6_model, delimiter=',')
np.savetxt('csv/Vout_all_SineSweep_10_6_test_model.txt', Vout_all_SineSweep_10_6_model, delimiter=',')
np.savetxt('csv/Iout_all_SineSweep_10_6_test_model.txt', Iout_all_SineSweep_10_6_model, delimiter=',')






#reconstruction and save checkpoint, save and plot results
runs_recon = 300
learning_rate_recon = 5e-3  #see paper





from reconstruction import recon_Vin_left
optimizer_recon_Vin_left, reconed_Vin_left, run_recon_Vin_left = recon_Vin_left(model, lossfn, runs_recon, learning_rate_recon, seq_len, device, scaler_Vin, Iin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iout_all_SineSweep_10_6)
save_path_reconstruction_Vin_left = 'model/reconstruction_Vin_left_checkpoint.pth'

reconed_Vin_left_plot = reconed_Vin_left[:, -1]
reconed_Vin_left_plot = reconed_Vin_left_plot.detach().numpy()
reconed_Vin_left_plot_rescale = scaler_Vin.inverse_transform(reconed_Vin_left_plot)


from reconstruction import save_checkpoint_reconstruction
save_checkpoint_reconstruction(model, optimizer_recon_Vin_left, save_path_reconstruction_Vin_left, run_recon_Vin_left, reconed_Vin_left)

from gen_seq_cpu import gen_seq_recon_Vin
reconed_Vin, reconed_Vin_Iin, reconed_Vin_Vout, reconed_Vin_Iout = gen_seq_recon_Vin(save_path_reconstruction_Vin_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)

np.savetxt('csv/Vin_all_SineSweep_10_6_reconed_left_side.txt', reconed_Vin_left_plot_rescale, delimiter=',')
np.savetxt('csv/Vin_all_SineSweep_10_6_finally_reconed.txt', reconed_Vin, delimiter=',')

plt.plot(reconed_Vin_left_plot_rescale, label="Vin_left_side")
plt.plot(reconed_Vin, label="Vin_reconstructed")
plt.plot(df_Vin_SineSweep_10_6, label="Vin_original")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Voltage Vin / V")
plt.show()




from reconstruction import recon_Iin_left
optimizer_recon_Iin_left, reconed_Iin_left, run_recon_Iin_left = recon_Iin_left(model, lossfn, runs_recon, learning_rate_recon, seq_len, device, scaler_Iin, Vin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iout_all_SineSweep_10_6)

save_path_reconstruction_Iin_left = 'model/reconstruction_Iin_left_checkpoint.pth'

reconed_Iin_left_plot = reconed_Iin_left[:, -1]
reconed_Iin_left_plot = reconed_Iin_left_plot.detach().numpy()
reconed_Iin_left_plot_rescale = scaler_Iin.inverse_transform(reconed_Iin_left_plot)

save_checkpoint_reconstruction(model, optimizer_recon_Iin_left, save_path_reconstruction_Iin_left, run_recon_Iin_left, reconed_Iin_left)

from gen_seq_cpu import gen_seq_recon_Iin
reconed_Iin_Vin, reconed_Iin, reconed_Iin_Vout, reconed_Iin_Iout = gen_seq_recon_Iin(save_path_reconstruction_Iin_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)

np.savetxt('csv/Iin_all_SineSweep_10_6_reconed_left_side.txt', reconed_Iin_left_plot_rescale, delimiter=',')
np.savetxt('csv/Iin_all_SineSweep_10_6_finally_reconed.txt', reconed_Iin, delimiter=',')

plt.plot(reconed_Iin_left_plot_rescale, label="Iin_left_side")
plt.plot(reconed_Iin, label="Iin_reconstructed")
plt.plot(df_Iin_SineSweep_10_6, label="Iin_original")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Current Iin / A")
plt.show()




from reconstruction import recon_Vout_left
optimizer_recon_Vout_left, reconed_Vout_left, run_recon_Vout_left = recon_Vout_left(model, lossfn, runs_recon, learning_rate_recon, seq_len, device, scaler_Vout, Vin_all_SineSweep_10_6, Iin_all_SineSweep_10_6, Iout_all_SineSweep_10_6)

save_path_reconstruction_Vout_left = 'model/reconstruction_Vout_left_checkpoint.pth'

reconed_Vout_left_plot = reconed_Vout_left[:, -1]
reconed_Vout_left_plot = reconed_Vout_left_plot.detach().numpy()
reconed_Vout_left_plot_rescale = scaler_Vout.inverse_transform(reconed_Vout_left_plot)

save_checkpoint_reconstruction(model, optimizer_recon_Vout_left, save_path_reconstruction_Vout_left, run_recon_Vout_left, reconed_Vout_left)

from gen_seq_cpu import gen_seq_recon_Vout
reconed_Vout_Vin, reconed_Vout_Iin, reconed_Vout, reconed_Vout_Iout = gen_seq_recon_Vout(save_path_reconstruction_Vout_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)

np.savetxt('csv/Vout_all_SineSweep_10_6_reconed_left_side.txt', reconed_Vout_left_plot_rescale, delimiter=',')
np.savetxt('csv/Vout_all_SineSweep_10_6_finally_reconed.txt', reconed_Vout, delimiter=',')

plt.plot(reconed_Vout_left_plot_rescale, label="Vout_left")
plt.plot(reconed_Vout, label="Vout_reconstructed")
plt.plot(df_Vout_SineSweep_10_6, label="Vout_original")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Voltage Vout / V")
plt.show()




from reconstruction import recon_Iout_left
optimizer_recon_Iout_left, reconed_Iout_left, run_recon_Iout_left = recon_Iout_left(model, lossfn, runs_recon, learning_rate_recon, seq_len, device, scaler_Iout, Vin_all_SineSweep_10_6, Iin_all_SineSweep_10_6, Vout_all_SineSweep_10_6)

save_path_reconstruction_Iout_left = 'model/reconstruction_Iout_left_checkpoint.pth'

reconed_Iout_left_plot = reconed_Iout_left[:, -1]
reconed_Iout_left_plot = reconed_Iout_left_plot.detach().numpy()
reconed_Iout_left_plot_rescale = scaler_Iout.inverse_transform(reconed_Iout_left_plot)

save_checkpoint_reconstruction(model, optimizer_recon_Iout_left, save_path_reconstruction_Iout_left, run_recon_Iout_left, reconed_Iout_left)

from gen_seq_cpu import gen_seq_recon_Iout
reconed_Iout_Vin, reconed_Iout_Iin, reconed_Iout_Vout, reconed_Iout = gen_seq_recon_Iout(save_path_reconstruction_Iout_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)

np.savetxt('csv/Iout_all_SineSweep_10_6_reconed_left_side.txt', reconed_Iout_left_plot_rescale, delimiter=',')
np.savetxt('csv/Iout_all_SineSweep_10_6_finally_reconed.txt', reconed_Iout, delimiter=',')

plt.plot(reconed_Iout_left_plot_rescale, label="Iout_left")
plt.plot(reconed_Iout, label="Iout_reconstructed")
plt.plot(df_Iout_SineSweep_10_6, label="Iout_original")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Current Iout / A")
plt.show()






runs_recon2 = 3000



#reconstruction of two features at the same time
from reconstruction import recon_Iin_Vout_left
optimizer_recon_Iin_Vout_left, reconed_Iin_twofeat_left, reconed_Vout_twofeat_left, run_recon_Iin_Vout_left = recon_Iin_Vout_left(model, lossfn, runs_recon2, learning_rate_recon, seq_len, device, scaler_Iin, scaler_Vout, Vin_all_SineSweep_10_6, Iout_all_SineSweep_10_6)

from reconstruction import save_checkpoint_reconstruction_twofeat
save_path_reconstruction_Iin_Vout_left = 'model/reconstruction_Iin_Vout_left_checkpoint.pth'

reconed_Iin_twofeat_left_plot = reconed_Iin_twofeat_left[:, -1]
reconed_Iin_twofeat_left_plot = reconed_Iin_twofeat_left_plot.detach().numpy()
reconed_Iin_twofeat_left_plot_rescale = scaler_Iin.inverse_transform(reconed_Iin_twofeat_left_plot)

reconed_Vout_twofeat_left_plot = reconed_Vout_twofeat_left[:, -1]
reconed_Vout_twofeat_left_plot = reconed_Vout_twofeat_left_plot.detach().numpy()
reconed_Vout_twofeat_left_plot_rescale = scaler_Vout.inverse_transform(reconed_Vout_twofeat_left_plot)


save_checkpoint_reconstruction_twofeat(model, optimizer_recon_Iin_Vout_left, save_path_reconstruction_Iin_Vout_left, run_recon_Iin_Vout_left, reconed_Iin_twofeat_left, reconed_Vout_twofeat_left)

from gen_seq_cpu import gen_seq_recon_Iin_Vout
reconed_Iin_Vout_twofeat_Vin, reconed_Iin_Vout_twofeat_Iin, reconed_Iin_Vout_twofeat_Vout, reconed_Iin_Vout_twofeat_Iout = gen_seq_recon_Iin_Vout(save_path_reconstruction_Iin_Vout_left, seq_len, inputsize, hiddensize1, hiddensize2, hiddensize3, hiddensize4, hiddensize5)




#save results
np.savetxt('csv/Iin_twofeat_all_SineSweep_10_6_reconed_left_side.txt', reconed_Iin_twofeat_left_plot_rescale, delimiter=',')
np.savetxt('csv/Iin_twofeat_all_SineSweep_10_6_finally_reconed.txt', reconed_Iin_Vout_twofeat_Iin, delimiter=',')

np.savetxt('csv/Vout_twofeat_all_SineSweep_10_6_reconed_left_side.txt', reconed_Vout_twofeat_left_plot_rescale, delimiter=',')
np.savetxt('csv/Vout_twofeat_all_SineSweep_10_6_finally_reconed.txt', reconed_Iin_Vout_twofeat_Vout, delimiter=',')




#plot results (training vs. reconstructed) 
plt.plot(reconed_Iin_twofeat_left_plot_rescale, label="Iin_twofeat_left")
plt.plot(reconed_Iin_Vout_twofeat_Iin, label="Iin_twofeat_reconstructed")
plt.plot(df_Iin_SineSweep_10_6, label="Iin_twofeat_original")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Current Iin / A")
plt.show()
plt.plot(reconed_Vout_twofeat_left_plot_rescale, label="Vout_twofeat_left")
plt.plot(reconed_Iin_Vout_twofeat_Vout, label="Vout_twofeat_reconstructed")
plt.plot(df_Vout_SineSweep_10_6, label="Vout_twofeat_original")
plt.legend(loc="upper right")
plt.xlabel("sample point")
plt.ylabel("Voltage Vout / V")
plt.show()




