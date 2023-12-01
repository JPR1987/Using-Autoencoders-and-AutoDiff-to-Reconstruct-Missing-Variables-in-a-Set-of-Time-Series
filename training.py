# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:27:18 2023

@author: roche
"""

import time
import torch

def run_training(runs, epoches, lossfn, optimizer, model, Vin_train_SineSweep_10_6, Vout_train_SineSweep_10_6, Vin_val_SineSweep_10_6, Vout_val_SineSweep_10_6, Vin_test_SineSweep_10_6, Vout_test_SineSweep_10_6, Vin_all_SineSweep_10_6, Vout_all_SineSweep_10_6, Iin_train_SineSweep_10_6, Iout_train_SineSweep_10_6, Iin_val_SineSweep_10_6, Iout_val_SineSweep_10_6, Iin_test_SineSweep_10_6, Iout_test_SineSweep_10_6, Iin_all_SineSweep_10_6, Iout_all_SineSweep_10_6 , Vin_train_DC_2b_6, Vout_train_DC_2b_6, Vin_val_DC_2b_6, Vout_val_DC_2b_6, Vin_test_DC_2b_6, Vout_test_DC_2b_6, Vin_all_DC_2b_6, Vout_all_DC_2b_6, Iin_train_DC_2b_6, Iout_train_DC_2b_6, Iin_val_DC_2b_6, Iout_val_DC_2b_6, Iin_test_DC_2b_6, Iout_test_DC_2b_6, Iin_all_DC_2b_6, Iout_all_DC_2b_6, Vin_train_DC_2c_8, Vout_train_DC_2c_8, Vin_val_DC_2c_8, Vout_val_DC_2c_8, Vin_test_DC_2c_8, Vout_test_DC_2c_8, Vin_all_DC_2c_8, Vout_all_DC_2c_8, Iin_train_DC_2c_8, Iout_train_DC_2c_8, Iin_val_DC_2c_8, Iout_val_DC_2c_8, Iin_test_DC_2c_8, Iout_test_DC_2c_8, Iin_all_DC_2c_8, Iout_all_DC_2c_8, Vin_train_DC_3a_6, Vout_train_DC_3a_6, Vin_val_DC_3a_6, Vout_val_DC_3a_6, Vin_test_DC_3a_6, Vout_test_DC_3a_6, Vin_all_DC_3a_6, Vout_all_DC_3a_6, Iin_train_DC_3a_6, Iout_train_DC_3a_6, Iin_val_DC_3a_6, Iout_val_DC_3a_6, Iin_test_DC_3a_6, Iout_test_DC_3a_6, Iin_all_DC_3a_6, Iout_all_DC_3a_6, Vin_train_Pulse_5b_9, Vout_train_Pulse_5b_9, Vin_val_Pulse_5b_9, Vout_val_Pulse_5b_9, Vin_test_Pulse_5b_9, Vout_test_Pulse_5b_9, Vin_all_Pulse_5b_9, Vout_all_Pulse_5b_9, Iin_train_Pulse_5b_9, Iout_train_Pulse_5b_9, Iin_val_Pulse_5b_9, Iout_val_Pulse_5b_9, Iin_test_Pulse_5b_9, Iout_test_Pulse_5b_9, Iin_all_Pulse_5b_9, Iout_all_Pulse_5b_9, Vin_train_Pulse_6c_5, Vout_train_Pulse_6c_5, Vin_val_Pulse_6c_5, Vout_val_Pulse_6c_5, Vin_test_Pulse_6c_5, Vout_test_Pulse_6c_5, Vin_all_Pulse_6c_5, Vout_all_Pulse_6c_5, Iin_train_Pulse_6c_5, Iout_train_Pulse_6c_5, Iin_val_Pulse_6c_5, Iout_val_Pulse_6c_5, Iin_test_Pulse_6c_5, Iout_test_Pulse_6c_5, Iin_all_Pulse_6c_5, Iout_all_Pulse_6c_5, Vin_train_PWM3, Vout_train_PWM3, Vin_val_PWM3, Vout_val_PWM3, Vin_test_PWM3, Vout_test_PWM3, Vin_all_PWM3, Vout_all_PWM3, Iin_train_PWM3, Iout_train_PWM3, Iin_val_PWM3, Iout_val_PWM3, Iin_test_PWM3, Iout_test_PWM3, Iin_all_PWM3, Iout_all_PWM3, Vin_train_DC_1b_6, Vout_train_DC_1b_6, Vin_val_DC_1b_6, Vout_val_DC_1b_6, Vin_test_DC_1b_6, Vout_test_DC_1b_6, Vin_all_DC_1b_6, Vout_all_DC_1b_6, Iin_train_DC_1b_6, Iout_train_DC_1b_6, Iin_val_DC_1b_6, Iout_val_DC_1b_6, Iin_test_DC_1b_6, Iout_test_DC_1b_6, Iin_all_DC_1b_6, Iout_all_DC_1b_6):
    
    for tr in range(runs):
        
        for t_DC_2b_6 in range(epoches):
            
            
            start_time = time.time()
            
            Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(Vin_train_DC_2b_6, Iin_train_DC_2b_6, Vout_train_DC_2b_6, Iout_train_DC_2b_6)
            Vin_pred_val, Iin_pred_val, Vout_pred_val, Iout_pred_val = model(Vin_val_DC_2b_6, Iin_val_DC_2b_6, Vout_val_DC_2b_6, Iout_val_DC_2b_6)
        
        
        
            # Compute and print loss.
            lossVin = lossfn(Vin_pred[1000: , :], Vin_train_DC_2b_6[1000: , :])
            lossIin = lossfn(Iin_pred[1000: , :], Iin_train_DC_2b_6[1000: , :])
            lossVout = lossfn(Vout_pred[1000: , :], Vout_train_DC_2b_6[1000: , :])
            lossIout = lossfn(Iout_pred[1000: , :], Iout_train_DC_2b_6[1000: , :])
            
            loss = lossVin + lossIin + lossVout + lossIout
            
            
            lossVin_val = lossfn(Vin_pred_val[1000: , :], Vin_val_DC_2b_6[1000: , :])
            lossIin_val = lossfn(Iin_pred_val[1000: , :], Iin_val_DC_2b_6[1000: , :])
            lossVout_val = lossfn(Vout_pred_val[1000: , :], Vout_val_DC_2b_6[1000: , :])
            lossIout_val = lossfn(Iout_pred_val[1000: , :], Iout_val_DC_2b_6[1000: , :])
            
            loss_val = lossVin_val + lossIin_val + lossVout_val + lossIout_val
            
            
            
        
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
        
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
        
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            elapsed = time.time() - start_time
            
            if (t_DC_2b_6+1)/(t_DC_2b_6+1) == 1:        
                print("Epoch_DC_2b_6:", t_DC_2b_6+1, "of", epoches, "Run:", tr+1, "of", runs, "\n Training Loss:", loss.item(),  "\n Training LossVin:", lossVin.item(), "\n Training LossIin:", lossIin.item(), "\n Training LossVout:", lossVout.item(), "\n Training LossIout:", lossIout.item())
                print(" Validation Loss:", loss_val.item(),  "\n Validation LossVin:", lossVin_val.item(), "\n Validation LossIin:", lossIin_val.item(), "\n Validation LossVout:", lossVout_val.item(), "\n Validation LossIout:", lossIout_val.item(), "\n Elapsed Time:", elapsed, "\n   ")

        
        
        
        
        
        for t_DC_2c_8 in range(epoches):
           
            
            start_time = time.time()
            
            Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(Vin_train_DC_2c_8, Iin_train_DC_2c_8, Vout_train_DC_2c_8, Iout_train_DC_2c_8)
            Vin_pred_val, Iin_pred_val, Vout_pred_val, Iout_pred_val = model(Vin_val_DC_2c_8, Iin_val_DC_2c_8, Vout_val_DC_2c_8, Iout_val_DC_2c_8)
        
        
        
            # Compute and print loss.
            lossVin = lossfn(Vin_pred[1000: , :], Vin_train_DC_2c_8[1000: , :])
            lossIin = lossfn(Iin_pred[1000: , :], Iin_train_DC_2c_8[1000: , :])
            lossVout = lossfn(Vout_pred[1000: , :], Vout_train_DC_2c_8[1000: , :])
            lossIout = lossfn(Iout_pred[1000: , :], Iout_train_DC_2c_8[1000: , :])
            
            loss = lossVin + lossIin + lossVout + lossIout
            
            
            lossVin_val = lossfn(Vin_pred_val[1000: , :], Vin_val_DC_2c_8[1000: , :])
            lossIin_val = lossfn(Iin_pred_val[1000: , :], Iin_val_DC_2c_8[1000: , :])
            lossVout_val = lossfn(Vout_pred_val[1000: , :], Vout_val_DC_2c_8[1000: , :])
            lossIout_val = lossfn(Iout_pred_val[1000: , :], Iout_val_DC_2c_8[1000: , :])
            
            loss_val = lossVin_val + lossIin_val + lossVout_val + lossIout_val
            
            
            
        
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
        
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
        
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            elapsed = time.time() - start_time
            
            if (t_DC_2c_8+1)/(t_DC_2c_8+1) == 1:        
                print("Epoch_DC_2c_8:", t_DC_2c_8+1, "of", epoches, "Run:", tr+1, "of", runs, "\n Training Loss:", loss.item(),  "\n Training LossVin:", lossVin.item(), "\n Training LossIin:", lossIin.item(), "\n Training LossVout:", lossVout.item(), "\n Training LossIout:", lossIout.item())
                print(" Validation Loss:", loss_val.item(),  "\n Validation LossVin:", lossVin_val.item(), "\n Validation LossIin:", lossIin_val.item(), "\n Validation LossVout:", lossVout_val.item(), "\n Validation LossIout:", lossIout_val.item(),"\n Elapsed Time:", elapsed, "\n          ")
                
        
        
        
        
        
        
        
        for t_DC_3a_6 in range(epoches):
           
            
            start_time = time.time()
            
            Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(Vin_train_DC_3a_6, Iin_train_DC_3a_6, Vout_train_DC_3a_6, Iout_train_DC_3a_6)
            Vin_pred_val, Iin_pred_val, Vout_pred_val, Iout_pred_val = model(Vin_val_DC_3a_6, Iin_val_DC_3a_6, Vout_val_DC_3a_6, Iout_val_DC_3a_6)
        
        
        
            # Compute and print loss.
            lossVin = lossfn(Vin_pred[1000: , :], Vin_train_DC_3a_6[1000: , :])
            lossIin = lossfn(Iin_pred[1000: , :], Iin_train_DC_3a_6[1000: , :])
            lossVout = lossfn(Vout_pred[1000: , :], Vout_train_DC_3a_6[1000: , :])
            lossIout = lossfn(Iout_pred[1000: , :], Iout_train_DC_3a_6[1000: , :])
            
            loss = lossVin + lossIin + lossVout + lossIout
            
            
            lossVin_val = lossfn(Vin_pred_val[1000: , :], Vin_val_DC_3a_6[1000: , :])
            lossIin_val = lossfn(Iin_pred_val[1000: , :], Iin_val_DC_3a_6[1000: , :])
            lossVout_val = lossfn(Vout_pred_val[1000: , :], Vout_val_DC_3a_6[1000: , :])
            lossIout_val = lossfn(Iout_pred_val[1000: , :], Iout_val_DC_3a_6[1000: , :])
            
            loss_val = lossVin_val + lossIin_val + lossVout_val + lossIout_val
            
            
            
        
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
        
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
        
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            elapsed = time.time() - start_time
            
            if (t_DC_3a_6+1)/(t_DC_3a_6+1) == 1:        
                print("Epoch_DC_3a_6:", t_DC_3a_6+1, "of", epoches, "Run:", tr+1, "of", runs, "\n Training Loss:", loss.item(),  "\n Training LossVin:", lossVin.item(), "\n Training LossIin:", lossIin.item(), "\n Training LossVout:", lossVout.item(), "\n Training LossIout:", lossIout.item())
                print(" Validation Loss:", loss_val.item(),  "\n Validation LossVin:", lossVin_val.item(), "\n Validation LossIin:", lossIin_val.item(), "\n Validation LossVout:", lossVout_val.item(), "\n Validation LossIout:", lossIout_val.item(),"\n Elapsed Time:", elapsed, "\n      ")
                





        for t_Pulse_5b_9 in range(epoches):
           
            
            start_time = time.time()
            
            Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(Vin_train_Pulse_5b_9, Iin_train_Pulse_5b_9, Vout_train_Pulse_5b_9, Iout_train_Pulse_5b_9)
            Vin_pred_val, Iin_pred_val, Vout_pred_val, Iout_pred_val = model(Vin_val_Pulse_5b_9, Iin_val_Pulse_5b_9, Vout_val_Pulse_5b_9, Iout_val_Pulse_5b_9)
        
        
        
            # Compute and print loss.
            lossVin = lossfn(Vin_pred[1000: , :], Vin_train_Pulse_5b_9[1000: , :])
            lossIin = lossfn(Iin_pred[1000: , :], Iin_train_Pulse_5b_9[1000: , :])
            lossVout = lossfn(Vout_pred[1000: , :], Vout_train_Pulse_5b_9[1000: , :])
            lossIout = lossfn(Iout_pred[1000: , :], Iout_train_Pulse_5b_9[1000: , :])
            
            loss = lossVin + lossIin + lossVout + lossIout
           
            
            lossVin_val = lossfn(Vin_pred_val[1000: , :], Vin_val_Pulse_5b_9[1000: , :])
            lossIin_val = lossfn(Iin_pred_val[1000: , :], Iin_val_Pulse_5b_9[1000: , :])
            lossVout_val = lossfn(Vout_pred_val[1000: , :], Vout_val_Pulse_5b_9[1000: , :])
            lossIout_val = lossfn(Iout_pred_val[1000: , :], Iout_val_Pulse_5b_9[1000: , :])
            
            loss_val = lossVin_val + lossIin_val + lossVout_val + lossIout_val
            
            
            
        
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
        
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
        
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            elapsed = time.time() - start_time
            
            if (t_Pulse_5b_9+1)/(t_Pulse_5b_9+1) == 1:        
                print("Epoch_Pulse_5b_9:", t_Pulse_5b_9+1, "of", epoches, "Run:", tr+1, "of", runs, "\n Training Loss:", loss.item(),  "\n Training LossVin:", lossVin.item(), "\n Training LossIin:", lossIin.item(), "\n Training LossVout:", lossVout.item(), "\n Training LossIout:", lossIout.item())
                print(" Validation Loss:", loss_val.item(),  "\n Validation LossVin:", lossVin_val.item(), "\n Validation LossIin:", lossIin_val.item(), "\n Validation LossVout:", lossVout_val.item(), "\n Validation LossIout:", lossIout_val.item(),"\n Elapsed Time:", elapsed, "\n     ")
               




        for t_Pulse_6c_5 in range(epoches):
            
            
            start_time = time.time()
            
            Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(Vin_train_Pulse_6c_5, Iin_train_Pulse_6c_5, Vout_train_Pulse_6c_5, Iout_train_Pulse_6c_5)
            Vin_pred_val, Iin_pred_val, Vout_pred_val, Iout_pred_val = model(Vin_val_Pulse_6c_5, Iin_val_Pulse_6c_5, Vout_val_Pulse_6c_5, Iout_val_Pulse_6c_5)
        
        
        
            # Compute and print loss.
            lossVin = lossfn(Vin_pred[1000: , :], Vin_train_Pulse_6c_5[1000: , :])
            lossIin = lossfn(Iin_pred[1000: , :], Iin_train_Pulse_6c_5[1000: , :])
            lossVout = lossfn(Vout_pred[1000: , :], Vout_train_Pulse_6c_5[1000: , :])
            lossIout = lossfn(Iout_pred[1000: , :], Iout_train_Pulse_6c_5[1000: , :])
            
            loss = lossVin + lossIin + lossVout + lossIout
            
            
            lossVin_val = lossfn(Vin_pred_val[1000: , :], Vin_val_Pulse_6c_5[1000: , :])
            lossIin_val = lossfn(Iin_pred_val[1000: , :], Iin_val_Pulse_6c_5[1000: , :])
            lossVout_val = lossfn(Vout_pred_val[1000: , :], Vout_val_Pulse_6c_5[1000: , :])
            lossIout_val = lossfn(Iout_pred_val[1000: , :], Iout_val_Pulse_6c_5[1000: , :])
            
            loss_val = lossVin_val + lossIin_val + lossVout_val + lossIout_val
            
            
            
        
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
        
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
        
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            elapsed = time.time() - start_time
            
            if (t_Pulse_6c_5+1)/(t_Pulse_6c_5+1) == 1:        
                print("Epoch_Pulse_6c_5:", t_Pulse_6c_5+1, "of", epoches, "Run:", tr+1, "of", runs, "\n Training Loss:", loss.item(),  "\n Training LossVin:", lossVin.item(), "\n Training LossIin:", lossIin.item(), "\n Training LossVout:", lossVout.item(), "\n Training LossIout:", lossIout.item())
                print(" Validation Loss:", loss_val.item(),  "\n Validation LossVin:", lossVin_val.item(), "\n Validation LossIin:", lossIin_val.item(), "\n Validation LossVout:", lossVout_val.item(), "\n Validation LossIout:", lossIout_val.item(),"\n Elapsed Time:", elapsed, "\n  ")
                





        for t_PWM3 in range(epoches):
           
            
            start_time = time.time()
            
            Vin_pred, Iin_pred, Vout_pred, Iout_pred = model.forward(Vin_train_PWM3, Iin_train_PWM3, Vout_train_PWM3, Iout_train_PWM3)
            Vin_pred_val, Iin_pred_val, Vout_pred_val, Iout_pred_val = model(Vin_val_PWM3, Iin_val_PWM3, Vout_val_PWM3, Iout_val_PWM3)
        
        
        
            # Compute and print loss.
            lossVin = lossfn(Vin_pred[1000: , :], Vin_train_PWM3[1000: , :])
            lossIin = lossfn(Iin_pred[1000: , :], Iin_train_PWM3[1000: , :])
            lossVout = lossfn(Vout_pred[1000: , :], Vout_train_PWM3[1000: , :])
            lossIout = lossfn(Iout_pred[1000: , :], Iout_train_PWM3[1000: , :])
            
            loss = lossVin + lossIin + lossVout + lossIout
           
            
            lossVin_val = lossfn(Vin_pred_val[1000: , :], Vin_val_PWM3[1000: , :])
            lossIin_val = lossfn(Iin_pred_val[1000: , :], Iin_val_PWM3[1000: , :])
            lossVout_val = lossfn(Vout_pred_val[1000: , :], Vout_val_PWM3[1000: , :])
            lossIout_val = lossfn(Iout_pred_val[1000: , :], Iout_val_PWM3[1000: , :])
            
            loss_val = lossVin_val + lossIin_val + lossVout_val + lossIout_val
            
            
            
        
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
        
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
        
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            elapsed = time.time() - start_time
            
            if (t_PWM3+1)/(t_PWM3+1) == 1:        
                print("Epoch_PWM3:", t_PWM3+1, "of", epoches, "Run:", tr+1, "of", runs, "\n Training Loss:", loss.item(),  "\n Training LossVin:", lossVin.item(), "\n Training LossIin:", lossIin.item(), "\n Training LossVout:", lossVout.item(), "\n Training LossIout:", lossIout.item())
                print(" Validation Loss:", loss_val.item(),  "\n Validation LossVin:", lossVin_val.item(), "\n Validation LossIin:", lossIin_val.item(), "\n Validation LossVout:", lossVout_val.item(), "\n Validation LossIout:", lossIout_val.item(),"\n Elapsed Time:", elapsed , "\n          ")
                
        
    return model, optimizer, t_PWM3+1, tr+1       
        
        
        
        
def save_checkpoint_training(model, optimizer, save_path, epoch, run):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'run': run
        }, save_path)
        
        
    
            


        


