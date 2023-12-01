# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:27:52 2023

@author: roche
"""

import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import time







def recon_Vin_left(model, lossfn, runs_app, learning_rate_app, seq_len, device, scaler_Vin, Iin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iout_all_SineSweep_10_6):
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var
    
    length=len(Iin_all_SineSweep_10_6)+seq_len
    
    ADVar1_init = Variable(torch.zeros(length, 1))
    ADVar1_scaled = scaler_Vin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True
    
    optimizer_app_onefeat = optim.Adam([ADVar1_seq], lr=learning_rate_app)
    
    
    A_app_all = ADVar1_seq.to(device)
    B_app_all = Iin_all_SineSweep_10_6.to(device)
    C_app_all = Vout_all_SineSweep_10_6.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)
    
    
    
    Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
    lossIin = lossfn(Iin_pred[1000: , :], B_app_all[1000: , :])
    lossVout = lossfn(Vout_pred[1000: , :], C_app_all[1000: , :])
    lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
    loss = lossIin + lossIout + lossVout

    loss_values = loss
    
    
    
    
    for t_app in range(runs_app):
                # Forward pass: compute predicted y by passing x to the model.
                                
                A_app_all = ADVar1_seq.to(device)
                B_app_all = Iin_all_SineSweep_10_6.to(device)
                C_app_all = Vout_all_SineSweep_10_6.to(device)
                D_app_all = Iout_all_SineSweep_10_6.to(device)
                
               
                           
                start_time = time.time()
                
                Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
               
            
                # Compute and print loss.
                lossIin = lossfn(Iin_pred[1000: , :], B_app_all[1000: , :])              
                lossVout = lossfn(Vout_pred[1000: , :], C_app_all[1000: , :])
                lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
                
                loss = lossIin + lossIout + lossVout
                
                
                
            
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer_app_onefeat.zero_grad()
            
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
            
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer_app_onefeat.step()
                elapsed = time.time() - start_time
                
                loss_values = torch.cat((loss_values.reshape(t_app + 1), loss.reshape(1)), 0)
                
                
                if (t_app+1)/(t_app+1) == 1:        
                    print("Reconstruction Run:", t_app+1, "of", runs_app, "\n Recon_Vin LossSum:", loss.item(),  "\n Recon_Vin LossIin:", lossIin.item(), "\n Recon_Vin LossVout:", lossVout.item(), "\n Recon_Vin LossIout:", lossIout.item(),"\n Elapsed Time:", elapsed)
                    
    
    print("Reconstruction of Vin finished.")                
                    
    return optimizer_app_onefeat, ADVar1_seq, t_app+1
          







def save_checkpoint_reconstruction(model, optimizer, save_path, run, ADVar1_seq):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'epoch': epoch,
        'run': run,
        'ADVar1_seq': ADVar1_seq 
        }, save_path)
    print("Reconstruction saved.")


def save_checkpoint_reconstruction_twofeat(model, optimizer, save_path, run, ADVar1_seq, ADVar2_seq):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'epoch': epoch,
        'run': run,
        'ADVar1_seq': ADVar1_seq,
        'ADVar2_seq':ADVar2_seq
        }, save_path)
    print("Reconstruction saved.")







def recon_Iin_left(model, lossfn, runs_app, learning_rate_app, seq_len, device, scaler_Iin, Vin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iout_all_SineSweep_10_6):
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var
    
    length=len(Vin_all_SineSweep_10_6) + seq_len
    
    ADVar1_init = Variable(torch.zeros(length, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True
    
    optimizer_app_onefeat = optim.Adam([ADVar1_seq], lr=learning_rate_app)
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = ADVar1_seq.to(device)
    C_app_all = Vout_all_SineSweep_10_6.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)
    
    
    
    Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
    lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
    lossVout = lossfn(Vout_pred[1000: , :], C_app_all[1000: , :])
    lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
    loss = lossVin + lossIout + lossVout

    loss_values = loss
    
    
    
    
    for t_app in range(runs_app):
                # Forward pass: compute predicted y by passing x to the model.
                
                
                A_app_all = Vin_all_SineSweep_10_6.to(device)
                B_app_all = ADVar1_seq.to(device)
                C_app_all = Vout_all_SineSweep_10_6.to(device)
                D_app_all = Iout_all_SineSweep_10_6.to(device)
                
               
                           
                start_time = time.time()
                
                Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
                
            
            
                # Compute and print loss.
                lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
                
                lossVout = lossfn(Vout_pred[1000: , :], C_app_all[1000: , :])
                lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
               
                loss = lossVin + lossIout + lossVout
               
                
            
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer_app_onefeat.zero_grad()
            
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
            
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer_app_onefeat.step()
                elapsed = time.time() - start_time
                
                loss_values = torch.cat((loss_values.reshape(t_app + 1), loss.reshape(1)), 0)
                
                
                if (t_app+1)/(t_app+1) == 1:        
                    print("Reconstruction Run:", t_app+1, "of", runs_app, "\n Recon_Iin LossSum:", loss.item(),  "\n Recon_Iin LossVin:", lossVin.item(), "\n Recon_Iin LossVout:", lossVout.item(), "\n Recon_Iin LossIout:", lossIout.item(),"\n Elapsed Time:", elapsed)
                    
                    
    print("Reconstruction of Iin finished.")
                    
   
                    
    return optimizer_app_onefeat, ADVar1_seq, t_app+1
          








def recon_Vout_left(model, lossfn, runs_app, learning_rate_app, seq_len, device, scaler_Vout, Vin_all_SineSweep_10_6, Iin_all_SineSweep_10_6, Iout_all_SineSweep_10_6):
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var
    
    length=len(Vin_all_SineSweep_10_6) + seq_len
    
    ADVar1_init = Variable(torch.zeros(length, 1))
    ADVar1_scaled = scaler_Vout.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True
    
    optimizer_app_onefeat = optim.Adam([ADVar1_seq], lr=learning_rate_app)
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = Iin_all_SineSweep_10_6.to(device)
    C_app_all = ADVar1_seq.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)
    
    
    
    Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
    lossIin = lossfn(Iin_pred[1000: , :], B_app_all[1000: , :])
    lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
    lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
    loss = lossIin + lossIout + lossVin

    loss_values = loss
    
    
    
    
    for t_app in range(runs_app):
                # Forward pass: compute predicted y by passing x to the model.
                
                
                A_app_all = Vin_all_SineSweep_10_6.to(device)
                B_app_all = Iin_all_SineSweep_10_6.to(device)
                C_app_all = ADVar1_seq.to(device)
                D_app_all = Iout_all_SineSweep_10_6.to(device)
                
               
                           
                start_time = time.time()
                
                Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
                
            
                # Compute and print loss.
                lossIin = lossfn(Iin_pred[1000: , :], B_app_all[1000: , :])                
                lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
                lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
               
                loss = lossIin + lossIout + lossVin
                
                
                
            
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer_app_onefeat.zero_grad()
            
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
            
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer_app_onefeat.step()
                elapsed = time.time() - start_time
                
                loss_values = torch.cat((loss_values.reshape(t_app + 1), loss.reshape(1)), 0)
                
                
                if (t_app+1)/(t_app+1) == 1:        
                    print("Reconstruction Run:", t_app+1, "of", runs_app, "\n Recon_Vout LossSum:", loss.item(), "\n Recon_Vout LossVin:", lossVin.item(),  "\n Recon_Vout LossIin:", lossIin.item(), "\n Recon_Vout LossIout:", lossIout.item(),"\n Elapsed Time:", elapsed)
                   
     
    print("Reconstruction of Vout finished.")               
     
           
           
    return optimizer_app_onefeat, ADVar1_seq, t_app+1
          











def recon_Iout_left(model, lossfn, runs_app, learning_rate_app, seq_len, device, scaler_Iout, Vin_all_SineSweep_10_6, Iin_all_SineSweep_10_6, Vout_all_SineSweep_10_6):
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var
    
    length=len(Vin_all_SineSweep_10_6) + seq_len
    
    ADVar1_init = Variable(torch.zeros(length, 1))
    ADVar1_scaled = scaler_Iout.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True
    
    optimizer_app_onefeat = optim.Adam([ADVar1_seq], lr=learning_rate_app)
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = Iin_all_SineSweep_10_6.to(device)
    C_app_all = Vout_all_SineSweep_10_6.to(device)
    D_app_all = ADVar1_seq.to(device)
    
    
    
    Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
    lossIin = lossfn(Iin_pred[1000: , :], B_app_all[1000: , :])
    lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
    lossVout = lossfn(Vout_pred[1000: , :], C_app_all[1000: , :])
    loss = lossIin + lossVout + lossVin

    loss_values = loss
    
    
    
    
    for t_app in range(runs_app):
                # Forward pass: compute predicted y by passing x to the model.
                
                
                A_app_all = Vin_all_SineSweep_10_6.to(device)
                B_app_all = Iin_all_SineSweep_10_6.to(device)
                C_app_all = Vout_all_SineSweep_10_6.to(device)
                D_app_all = ADVar1_seq.to(device)
                
               
                           
                start_time = time.time()
                
                Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
                
            
                # Compute and print loss.
                lossIin = lossfn(Iin_pred[1000: , :], B_app_all[1000: , :])
                lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
                lossVout = lossfn(Vout_pred[1000: , :], C_app_all[1000: , :])
                loss = lossIin + lossVout + lossVin
               
                
                
            
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer_app_onefeat.zero_grad()
            
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
            
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer_app_onefeat.step()
                elapsed = time.time() - start_time
                
                loss_values = torch.cat((loss_values.reshape(t_app + 1), loss.reshape(1)), 0)
                
                
                if (t_app+1)/(t_app+1) == 1:        
                    print("Reconstruction Run:", t_app+1, "of", runs_app, "\n Recon_Iout LossSum:", loss.item(), "\n Recon_Iout LossVin:", lossVin.item(),  "\n Recon_Iout LossIin:", lossIin.item(), "\n Recon_Iout LossVout:", lossVout.item(),"\n Elapsed Time:", elapsed)
                   
    
    print("Reconstruction of Iout finished.")                
    
    
                    
    return optimizer_app_onefeat, ADVar1_seq, t_app+1











def recon_Iin_Vout_left(model, lossfn, runs_app, learning_rate_app, seq_len, device, scaler_Iin, scaler_Vout, Vin_all_SineSweep_10_6, Iout_all_SineSweep_10_6):
    
    def transform_data(arr, seq_len):
        x = []
        for i in range(len(arr) - seq_len):
            x_i = arr[i : i + seq_len]
            x.append(x_i)
    #    x_arr = np.array(x).reshape(seq_len, -1, 1)
        x_arr = np.array(x).reshape(-1, seq_len, 1)
        x_var = Variable(torch.from_numpy(x_arr).float())
        return x_var
    
    length=len(Vin_all_SineSweep_10_6) + seq_len
    
    ADVar1_init = Variable(torch.zeros(length, 1))
    ADVar1_scaled = scaler_Iin.transform(ADVar1_init)
    ADVar1_seq = transform_data(ADVar1_scaled, seq_len)
    ADVar1_seq.requires_grad = True
       
    ADVar2_init = Variable(torch.zeros(length, 1))
    ADVar2_scaled = scaler_Vout.transform(ADVar2_init)
    ADVar2_seq = transform_data(ADVar2_scaled, seq_len)
    ADVar2_seq.requires_grad = True
    
    optimizer_app_twofeat = optim.Adam([ADVar1_seq, ADVar2_seq], lr=learning_rate_app)
    
    
    A_app_all = Vin_all_SineSweep_10_6.to(device)
    B_app_all = ADVar1_seq.to(device)
    C_app_all = ADVar2_seq.to(device)
    D_app_all = Iout_all_SineSweep_10_6.to(device)
    
    
    
    Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
    lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
    lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
    loss = lossIout + lossVin

    loss_values = loss
    
    
    
    
    for t_app in range(runs_app):
                # Forward pass: compute predicted y by passing x to the model.
                
                
                A_app_all = Vin_all_SineSweep_10_6.to(device)
                B_app_all = ADVar1_seq.to(device)
                C_app_all = ADVar2_seq.to(device)
                D_app_all = Iout_all_SineSweep_10_6.to(device)
                
               
                           
                start_time = time.time()
                
                Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(A_app_all, B_app_all, C_app_all, D_app_all)
                
            
            
                # Compute and print loss.
               
                lossVin = lossfn(Vin_pred[1000: , :], A_app_all[1000: , :])
                lossIout = lossfn(Iout_pred[1000: , :], D_app_all[1000: , :])
               
                loss = lossIout + lossVin
                
                
            
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer_app_twofeat.zero_grad()
            
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
            
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer_app_twofeat.step()
                elapsed = time.time() - start_time
                
                loss_values = torch.cat((loss_values.reshape(t_app + 1), loss.reshape(1)), 0)
                
                
                if (t_app+1)/(t_app+1) == 1:        
                    print("Reconstruction Run:", t_app+1, "of", runs_app, "\n Recon_Iin_Vout LossSum:", loss.item(), "\n Recon_Iin_Vout LossVin:", lossVin.item(),  "\n Recon_Iin_Vout LossIout:", lossIout.item(),"\n Elapsed Time:", elapsed)
                   
    
    print("Reconstruction of Iout finished.")                
    
    
                    
    return optimizer_app_twofeat, ADVar1_seq, ADVar2_seq, t_app+1






