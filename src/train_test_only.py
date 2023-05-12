"""
File Name : sl_animals_decolle
Author: Schechter

Creation Date : Mar 23. 2023

Adapted from the original code by : Emre Neftci
From the paper: "DECOLLE: Deep Continuous Local Learning"
"""
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter 
from decolle1.init_functions import init_LSUV
from decolle1.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayer #, LIFLayerVariableTau
from decolle1.utils import (parse_args, train, test, write_stats, #accuracy,
                            save_checkpoint, load_model_from_checkpoint, 
                            prepare_experiment_custom, cross_entropy_one_hot)
from dataset import AnimalsDvsSliced
from decolle_tools import (get_params, kfold_split, get_loss, get_optimizer, 
                           slice_data)


#assert we are on the right working directory
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#set printing option
np.set_printoptions(precision=4)

#load main parameters
#********************** choose one of 4 options ******************************
# params_file = 'parameters/params.yml'              #default 3 layers
# params_file = 'parameters/params_nmnist.yml'       #Nmnist 4 layers
# params_file = 'parameters/params_dvsgestures.yml'  #DVS Gestures 3 layers
params_file = 'parameters/params_slanimals.yml'      #SL-Animals 4 layers
#*****************************************************************************
params = get_params(params_file)

#set initial arguments and variables
args = parse_args(params_file)
args.save_dir=params['dataset']
starting_epoch = 0
best_losses, best_accuracies = [], []  #initialize best loss and acc history
device = args.device                   #default = 'cuda'
seed = args.seed                       #default = 0

#set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#creating a generator to split the data into 4 folds of train/test files
train_test_generator = kfold_split(params['file_list'], seed)


#this syntax is necessary to work with multiprocessing!
if __name__ == '__main__':
    
    #define directories for logs and results
    dirs = prepare_experiment_custom(name=os.path.basename(__file__).split('.')[0], 
                                     args = args)
    logs_dir = dirs['tensorbrd_dir'] 
    checkpoint_root_dir = dirs['checkpoint_dir']
    output_dir = dirs['output_dir']
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    #check if dataset is already sliced (else, slice the raw data files)
    slice_data(params['data_path'], params['file_list'])
    
    #print header
    print('\nWELCOME TO DECOLLE TRAINING!')
    print('Starting 4-fold cross validation (train/test only):\n')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        print("FOLD {}:".format(fold))
        print("-----------------------------------------------")
        
        #initialize histories
        test_acc_hist, test_loss_hist, train_loss_hist = [], [], []
    
        #logging statistics with tensorboard, update directories
        writer = SummaryWriter(os.path.join(logs_dir, 'fold{}'.format(fold)))
        checkpnt_dir = os.path.join(checkpoint_root_dir, 'fold{}'.format(fold))

        #definining train and test Datasets
        training_set = AnimalsDvsSliced(
            dataPath     = params['data_path'],
            fileList     = train_set,
            samplingTime = params['deltat']/1000,  #in ms
            sampleLength = params['chunk_size_train'],
            randomCrop   = True,
            redFactor    = 0.25,  #reduction of input size from 128 to 32
            binMode      = 'SUM'  #'OR'
        )
        testing_set = AnimalsDvsSliced(
            dataPath     = params['data_path'],
            fileList     = test_set,
            samplingTime = params['deltat']/1000,  #in ms
            sampleLength = params['chunk_size_test'],
            randomCrop   = False, 
            redFactor    = 0.25,  #reduction of input size from 128 to 32
            binMode      = 'SUM'  #'OR'
        )
        
        #definining train and test DataLoaders
        train_loader = torch.utils.data.DataLoader(
            dataset=training_set, 
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=params['num_dl_workers'])
        test_loader = torch.utils.data.DataLoader(
            dataset=testing_set, 
            batch_size=params['batch_size'],
            shuffle=False, 
            num_workers=params['num_dl_workers'])
    
        ## Create Model, Optimizer and Loss
        net = LenetDECOLLE( out_channels=params['out_channels'],
                            Nhid=params['Nhid'],
                            Mhid=params['Mhid'],
                            kernel_size=params['kernel_size'],
                            pool_size=params['pool_size'],
                            input_shape=params['input_shape'],
                            alpha=params['alpha'],
                            alpharp=params['alpharp'],
                            dropout=params['dropout'],
                            beta=params['beta'],
                            num_conv_layers=params['num_conv_layers'],
                            num_mlp_layers=params['num_mlp_layers'],
                            lc_ampl=params['lc_ampl'],
                            lif_layer_type = LIFLayer,
                            method=params['learning_method'],
                            with_output_layer=params['with_output_layer']).to(device)

        #optimizer
        if hasattr(params['learning_rate'], '__len__'):
            from decolle1.utils import MultiOpt
            opts = []
            for i in range(len(params['learning_rate'])):
                opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i], betas=params['betas']))
            opt = MultiOpt(*opts)
        else:  #working here
            opt = get_optimizer(net, params)
        
        #regularizer parameters
        reg_l = params['reg_l'] if 'reg_l' in params else None
        
        #loss
        if 'loss_scope' in params and params['loss_scope']=='global':
            loss = [None for i in range(len(net))]
            if net.with_output_layer: 
                loss[-1] = cross_entropy_one_hot
            else:
                raise RuntimeError('bptt mode needs output layer')
            decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)
        else:  #working here
            loss = [get_loss(params['loss']) for i in range(len(net))]
            if net.with_output_layer:
                loss[-1] = cross_entropy_one_hot
            decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)
        
        # #**********************************************************
        # if fold==1:  #DELETE LATER
        #     print('Single train/test fold training for now.') 
        # #**********************************************************
        
        #Initialize the network, if not resuming from a checkpoint
        if args.resume_from is None:
            """
            This step consumes a lot of memory! 
            In order to keep pool1 layer = 1x1 like in the SL-Animals paper, you may 
            need to reduce the number of samples used for initialization of the network 
            so the Tensor in the init routine fits in your memory. 
            Alternatively, you may continue using 32 samples like in the original init
            implementation, but you may need to change the network layer pool1 to 2x2; 
            it all comes down to your GPU memory size.
            """
            #get first training batch for DECOLLE initialization
            print("Getting 1st training batch for weight initialization, please wait...")
            data_batch, target_batch = next(iter(train_loader))
            data_batch = torch.Tensor(data_batch).to(device)
            target_batch = torch.Tensor(target_batch).to(device)
            #data shape: input [72, 500, 2, 32, 32] target [72, 500, 11]  #DVS-Gestures
            #data shape: input [76, 500, 2, 32, 32] target [76, 500, 19]  #SL-Animals
            print('DATA SHAPE:', data_batch.shape)
            print('TARGET SHAPE:', target_batch.shape)
            #use first 32 samples from batch for initialization
            init_samples = 32 if params['pool_size'][0] == 2 else 8
            net.init_parameters(data_batch[:init_samples])  
            print('\nInitializing network parameters:')
            init_LSUV(net, data_batch[:init_samples])
        else:  #resume training
            print("Checkpoint directory " + checkpnt_dir)
            if not os.path.exists(checkpnt_dir) and not args.no_save:
                os.makedirs(checkpnt_dir)
            starting_epoch = load_model_from_checkpoint(checkpnt_dir, net, opt)
            print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))
        
        # Printing parameters
        if args.verbose:
            print('\nUsing the following parameters:')
            m = max(len(x) for x in params)
            for k, v in zip(params.keys(), params.values()):
                print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))
        
        # --------TRAINING LOOP----------
        print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

        if not args.no_train:
            for e in range(starting_epoch , params['num_epochs'] ):
               
                #learning rate adjustment (every 'lr_drop_interval' epochs)
                actual_lr = opt.param_groups[-1]['lr']
                if (e % params['lr_drop_interval']) == 0 and e!=0:  #if interval
                    next_lr = actual_lr / params['lr_drop_factor']  #calc. next LR
                    opt.param_groups[-1]['lr'] = next_lr            #update LR
                    print('\n-> Changing learning rate from {} to {}'
                          .format(actual_lr, next_lr))
                else:  #maintain learning rate
                    print('\n-> Current learning rate: {}'.format(actual_lr))
    
                #train the network
                train_loss, fire_rate = train(
                    train_loader, decolle_loss, net, opt, e, 
                    params['burnin_steps'], 
                    online_update=params['online_update']
                    )
                #log train_loss history
                train_loss_hist.append(train_loss)
                #log firing rate to tensorboard(each layer)
                if not args.no_save:
                    for i in range(len(net)):
                        writer.add_scalar('/Firing_rate/{0}'.format(i), 
                                          fire_rate[i], e)
                
                #test the network
                test_loss, test_acc = test(
                    test_loader, decolle_loss, net, params['burnin_steps'], 
                    print_error = False
                    )
                #log test_loss and test_accuracy to history
                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                #log statistics to tensorboard (accuracy and loss)
                if not args.no_save:
                    write_stats(e, test_acc, test_loss, writer)
                    np.save(os.path.join(output_dir, 
                        'test_acc_fold{}.npy'.format(fold)), 
                        np.array(test_acc_hist))
                    np.save(os.path.join(output_dir, 
                        'test_loss_fold{}.npy'.format(fold)),
                        np.array(test_loss_hist))
                
                #print min. test loss / max. test accuracy
                print('(Max accuracy: {:.2f}%  | Min. loss: {:.2f})'.format(
                    100 * np.array(test_acc_hist).max(),
                    np.array(test_loss_hist).min()))
               
                #saving checkpoint
                if (e % params['save_interval']) == 0 and e!=0:
                    print('---------------Epoch {}-------------'.format(e))
                    if not args.no_save:
                        print('---------Saving checkpoint---------')
                        save_checkpoint(e, checkpnt_dir, net, opt)
               
                #end of this epoch
            #end of all epochs
       
            #plot results
            #accuracy
            print('Test Accuracy History (fold {}):'.format(fold))
            plt.figure()
            plt.plot(test_acc_hist)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.title('Test Accuracy on all layers', fontweight="bold", fontsize=14)
            plt.legend(["layer {}".format(l) for l in range(1, params['num_layers']+1)])
            plt.grid(visible=True, axis='y')
            # plt.ylim((0.0, 1.0))
            plt.savefig(os.path.join(output_dir, 
                                    'accuracy_fold{}.png'.format(fold)))
            #loss
            print('Train/Test Loss History (fold {}):'.format(fold))
            plt.figure()
            plt.plot(train_loss_hist, ':')
            plt.gca().set_prop_cycle(None)  #reset color cycle
            plt.plot(test_loss_hist)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.title('Loss on all layers', fontweight="bold", fontsize=14)
            plt.legend(["train layer {}".format(l) for l in range(1, params['num_layers']+1)] 
                       + ["test layer {}".format(l) for l in range(1, params['num_layers']+1)])
            plt.grid(visible=True, axis='y')
            plt.savefig(os.path.join(output_dir, 
                                    'loss_fold{}.png'.format(fold)))
            
            #print best results
            print('Best Results - fold {}:'.format(fold))
            print('----------------------')
            min_loss = np.array(test_loss_hist).min()
            max_acc = np.array(test_acc_hist).max()
            for l in range(1, params['num_layers']+1):
                print('Max Test Accuracy on Layer {}:  {:.2f}%'.format(l, 
                      (100 * max(np.array(test_acc_hist)[:, l-1]))))     
        # --------END OF TRAINING LOOP----------
            
        #save this fold's best losses and accuracies in history
        best_losses.append(min_loss)
        best_accuracies.append(max_acc)
            
        #end of IF FOLD==1
           
        writer.close()
        #end of fold
    #end of cross-validation
    
    global_end_time = datetime.now()     #monitor total training time
    print('\nGlobal Training Time:', global_end_time - global_st_time)
   
    #print final results
    print("\nMin Test Loss on 4 folds:", best_losses)
    print("Min Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(best_losses), np.std(best_losses)))

    print("\nMax Test Accuracy on 4 folds:", best_accuracies)
    print("Max Test Accuracy:     {:.2f}% +- {:.2f}%".format(
        100 * np.mean(best_accuracies), 100 * np.std(best_accuracies)))
        
    
#LOG OF RESULTS
#DVS-Gestures: 95.14% Max Test Accuracy
        
        
          
