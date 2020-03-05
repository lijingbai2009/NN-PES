### Neural Network for Photochemical Reaction Prediction
### Version 0.0 Jingbai Li Nov 13 2019
### Version 0.1 Jingbai Li Nov 19 2019 add hybrid training module NNEG
### Version 0.2 Jingbai Li Nov 23 2019 reconstruct the code structure
### Version 0.3 Jingbai Li Dec  2 2019 minor fix, imporve random search protocal
import time,datetime,os,sys,json
from optparse import OptionParser
import numpy as np
usage="""
 *--------------------------------------------------------------*
 |                                                              |
 |   Neural Network for Prediciton of Photochemical Reaction    |
 |                                                              |
 *--------------------------------------------------------------*

Usage:
  python3 NN-ChemI.py --td in_data [additional options]
  python3 NN-ChemI.py -h for more options
"""
description=''
parser = OptionParser(usage=usage, description=description)
parser.add_option('--td', dest='in_data',   type=str,   nargs=1, help='Training data in json format')
parser.add_option('--pd', dest='pred_data', type=str,   nargs=1, help='Prediction data in json format')
parser.add_option('--sl', dest='silent',    type=int,   nargs=1, help='=0 silent mode; =1 print verbose information; Default=0',default=0)
parser.add_option('--gs', dest='gl_seed',   type=int,   nargs=1, help='Global random seed; Defualt=0',default=0)
parser.add_option('--iw', dest='in_weight', type=int,   nargs=1, help='Neural Network modes: -2 hyper parameter search, requres Talos; -1 predict properties; 0 new train; >0 load trained weights',default=0)
parser.add_option('--st', dest='stat',      type=int,   nargs=1, help='Plot statistics of in_data, requres matplotlib; Defualt=0',default=0)
parser.add_option('--nn', dest='model_name',type=str,   nargs=1, help='Type of neural network; eg - energy+gradient; nac - non-adiabatic coupling; e - energy; g - gradient', default='eg')
parser.add_option('--ep', dest='ep',        type=int,   nargs=1, help='Epoch; Default=1',default=1)
parser.add_option('--bs', dest='bs',        type=int,   nargs=1, help='Batch size; Default=1',default=1)
parser.add_option('--hl', dest='nlayer',    type=int,   nargs=1, help='Hidden layer; Default=1',default=1)
parser.add_option('--nd', dest='node',      type=int,   nargs=1, help='Node per hidden layer; Default=1',default=1)
parser.add_option('--l2', dest='wl2',       type=float, nargs=1, help='L2 regularization rate; Default=1e-9',default=1e-9)
parser.add_option('--lr', dest='lr',        type=float, nargs=1, help='Learning rate; Default=3e-3',default=3e-3)
parser.add_option('--dl', dest='flr',       type=float, nargs=1, help='Learning rate decay factor; Default=0.9',default=0.9)
parser.add_option('--ds', dest='flrstep',   type=int,   nargs=1, help='Learning rate decay factor waiting step; Default=10',default=10)
parser.add_option('--NS', dest='nsample',   type=float, nargs=1, help='Random search sample ratio, iteration=ratio*search_space_size; Default=0.1',default=0.1)
parser.add_option('--EP', dest='s_ep',      type=int,   nargs=3, help='Random search epoch, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--BS', dest='s_bs',      type=int,   nargs=3, help='Random search batch size, requires initial, last and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--HL', dest='s_nlayer',  type=int,   nargs=3, help='Random search hidden layer, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--ND', dest='s_node',    type=int,   nargs=3, help='Random search node, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--LL', dest='s_wl2',     type=float, nargs=3, help='Random search L2 regularization rate, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--LR', dest='s_lr',      type=float, nargs=3, help='Random search learning rate, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--DL', dest='s_flr',     type=float, nargs=3, help='Random search learning rate decay factor, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--DS', dest='s_flrstep', type=int,   nargs=3, help='Random search learning rate decay waiting step, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
parser.add_option('--NI', dest='s_iter',    type=int,   nargs=1, help='Random search iteractions; Default=1',default=1)
parser.add_option('--WN', dest='s_win',     type=int,   nargs=1, help='Random search candidates; Defualt=4',default=4)

(options, args) = parser.parse_args()
if options.in_data == None:
    print (usage)
    exit()
np.random.seed(options.gl_seed)   # fix the random seed befor import keras
import keras
import keras.backend as K
from keras.losses import mse
from keras import regularizers
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import talos as ta
from talos import Scan

def whatistime():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def howlong(start,end):
    walltime=end-start
    walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
    return walltime

def params_space(space,var,par):
    #span hyperparameter space
    if space[-1] == 0:
        params=[var]
    else:
        params=[]
        start,end,step=space
        for i in range(step+1):
            if   par == 'ep' or par == 'nlayer' or par == 'node' or par == 'flrstep':      # (last-initial)/step
                p=start+(end-start)*(i/step)
                p=int(p)
            elif par == 'bs' or par == 'wl2' or par == 'lr' or par == 'flr':     # (last/initial)**(1/step)
                p=start*(end/start)**(i/step)
                if par == 'bs':
                    p=int(p)
            if p not in params:
                params.append(p)
    return params

def update_params(candidates,space,par):
    start,end,step=space
    if step !=0:
        index={'bs':-10,'ep':-9,'flr':-8,'flrstep':-7,'lr':-5,'nlayer':-4,'node':-3,'wl2':-1}
        candidates=candidates[:,index[par]]
        max=np.amax(candidates)
        min=np.amin(candidates)
        if par == 'bs' or par == 'ep' or par == 'nlayer' or par == 'node' or par == 'flrstep':
            max=int(max)
            min=int(min)
        if max == min:
            space=[max,max,1]
        else:
            space=[min,max,step]

    return space

def partition(sd,data):
    s=sd
    size=len(data)
    full=np.arange(size)
    weight_train=int(0.9*size)
    weight_validation=int(0.1*size)

    np.random.seed(s)
    pick_train=np.random.choice(full,weight_train,replace=False)
    remain=[i for i in full if i not in pick_train]
    np.random.seed(s)
    pick_validation=np.random.choice(remain,weight_validation,replace=False)
    pick_test=[i for i in remain if i not in pick_validation]

    train=data[pick_train,:]
    validation=data[pick_validation,:]
    test=data[pick_test,:]

    return train,validation,test

def lr_scheduler(epoch,lr):
    #flrstep and flr are global varable
    if (epoch+1) % flrstep == 0:
        lr=lr*flr
    return lr

def shifted_softplus(x):
    return K.log(0.5*K.exp(x)+0.5)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

def dmax(y_true,y_pred):
    return K.max(K.abs(y_pred -y_true))

def Record(model_name,arch,hist):
    if model_name == 'eg':
        output='%s\n' % (arch)
        output+='                 --- Training History ---\n'
        output+=' --------------------------------------------------------------\n'
        output+='Para%6s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n' % ('epoch','lr','loss','val_loss','mae','val_mae','rmse','val_rmse','dmax','val_dmax')
        n=len(hist['lr'])
        print(n)
        for i in range(n):
            output+='Ep: %6d%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % ((i+1),hist['lr'][i],hist['e_loss'][i],hist['val_e_loss'][i],hist['e_mae'][i],hist['val_e_mae'][i],hist['e_rmse'][i],hist['val_e_rmse'][i],hist['e_dmax'][i],hist['val_e_dmax'][i])
        output+='\n                 --- Training History ---\n'
        output+=' --------------------------------------------------------------\n'
        output+='Para%6s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n' % ('epoch','lr','loss','val_loss','mae','val_mae','rmse','val_rmse','dmax','val_dmax')
        n=len(hist['lr'])
        for i in range(n):
            output+='Ep: %6d%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % ((i+1),hist['lr'][i],hist['g_loss'][i],hist['val_g_loss'][i],hist['g_mae'][i],hist['val_g_mae'][i],hist['g_rmse'][i],hist['val_g_rmse'][i],hist['g_dmax'][i],hist['val_g_dmax'][i])
        output+='\n'
        log=open('model.log','a')
        log.write(output)
        log.close()
        result='%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % (hist['lr'][-1],hist['e_loss'][-1],hist['val_e_loss'][-1],hist['e_mae'][-1],hist['val_e_mae'][-1],hist['e_rmse'][-1],hist['val_e_rmse'][-1],hist['e_dmax'][-1],hist['val_e_dmax'][-1])
        result+='%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % (hist['lr'][-1],hist['g_loss'][-1],hist['val_g_loss'][-1],hist['g_mae'][-1],hist['val_g_mae'][-1],hist['g_rmse'][-1],hist['val_g_rmse'][-1],hist['g_dmax'][-1],hist['val_g_dmax'][-1])
        fin=open('model.sum','a')
        fin.write(result)
        fin.close()
    else:
        output='%s\n' % (arch)
        output+='                 --- Training History ---\n'
        output+=' --------------------------------------------------------------\n'
        output+='Para%6s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n' % ('epoch','lr','loss','val_loss','mae','val_mae','rmse','val_rmse','dmax','val_dmax')
        n=len(hist['lr'])
        for i in range(n):
            output+='Ep: %6d%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % ((i+1),hist['lr'][i],hist['loss'][i],hist['val_loss'][i],hist['mae'][i],hist['val_mae'][i],hist['rmse'][i],hist['val_rmse'][i],hist['dmax'][i],hist['val_dmax'][i],)
        output+='\n'
        log=open('model.log','a')
        log.write(output)
        log.close()
        result='%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % (hist['lr'][-1],hist['loss'][-1],hist['val_loss'][-1],hist['mae'][-1],hist['val_mae'][-1],hist['rmse'][-1],hist['val_rmse'][-1],hist['dmax'][-1],hist['val_dmax'][-1])
        fin=open('model.sum','a')
        fin.write(result)
        fin.close()

def Statistics(invr,energy,gradient,nac):
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import matplotlib as mpl

    fig,ax=plt.subplots(2,2)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)

    ax[0,0].set_title('invR (1/Angstrom)')
    ax[0,0].axes.tick_params(axis='both',direction='in',length=0)
    ax[0,0].set_xlim(-1.1,1.1)
    ax[0,0].hist(np.ndarray.flatten(invr),color='black',bins=200)

    ax[0,1].set_title('Energy (a.u.)')
    ax[0,1].axes.tick_params(axis='both',direction='in',length=0)
    ax[0,1].set_xlim(-1.1,1.1)
    ax[0,1].hist(np.ndarray.flatten(energy),color='black',bins=200)

    ax[1,0].set_title('Gradient (a.u.)')
    ax[1,0].axes.tick_params(axis='both',direction='in',length=0)
    ax[1,0].set_xlim(-1.1,1.1)
    ax[1,0].hist(np.ndarray.flatten(gradient),color='black',bins=400)

    ax[1,1].set_title('NAC (a.u.)')
    ax[1,1].axes.tick_params(axis='both',direction='in',length=0)
    ax[1,1].set_xlim(-1.1,1.1)
    ax[1,1].hist(np.ndarray.flatten(nac),color='black',bins=400)
    plt.savefig('model-stat.png',dpi=400)

def Prepdata(data,silent,gl_seed,stat):
    data_info=''
    natom,nstate,invr,energy,gradient,nac=data
    invr=np.array(invr)
    energy=np.array(energy)
    gradient=np.array(gradient)
    nac=np.array(nac)

    nmol=len(invr)                    # number of molecule
    ninvr=len(invr[0])                # number of distance per molecule, which is the input size
    nenergy=len(energy[0])            # number of energy per molecule, which is the output size
    ngrad=len(gradient[0])/(natom*3)  # number of gradient matrix per molecule
    nnac=len(nac[0])/(natom*3)        # number of non-adiabatic matrix per molecule

    max_invr=np.amax(invr)
    min_invr=np.amin(invr)
    mid_invr=(max_invr+min_invr)/2
    dev_invr=(max_invr-min_invr)/2
    avg_invr=np.mean(invr)
    std_invr=np.std(invr)
    miu_invr=mid_invr
    sgm_invr=dev_invr

    max_energy=np.amax(energy)
    min_energy=np.amin(energy)
    mid_energy=(max_energy+min_energy)/2
    dev_energy=(max_energy-min_energy)/2
    avg_energy=np.mean(energy)
    std_energy=np.std(energy)
    miu_energy=mid_energy
    sgm_energy=dev_energy

    max_gradient=np.amax(gradient)
    min_gradient=np.amin(gradient)
    mid_gradient=(max_gradient+min_gradient)/2
    dev_gradient=(max_gradient-min_gradient)/2
    avg_gradient=np.mean(gradient)
    std_gradient=np.std(gradient)
    miu_gradient=mid_gradient
    sgm_gradient=dev_gradient

    max_nac=np.amax(nac)
    min_nac=np.amin(nac)
    mid_nac=(max_nac+min_nac)/2
    dev_nac=(max_nac-min_nac)/2
    avg_nac=np.mean(nac)
    std_nac=np.std(nac)
    miu_nac=mid_nac
    sgm_nac=dev_nac

    data_info+="""
                  --- Data info ---
 --------------------------------------------------------------
    dist     max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
    energy   max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
    gradient max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
    nac      max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
""" % (np.amax(invr),np.amin(invr),avg_invr,std_invr,np.amax(energy),np.amin(energy),avg_energy,std_energy,np.amax(gradient),np.amin(gradient),avg_gradient,std_gradient,np.amax(nac),np.amin(nac),avg_nac,std_nac)

    # shift input to the averaged value and scaled by standard deviation
    invr=(invr-miu_invr)/sgm_invr
    energy=(energy-miu_energy)/sgm_energy
    gradient=(gradient-miu_gradient)/sgm_gradient
    nac=(nac-miu_nac)/sgm_nac
    data_info+="""
                  --- Preprocessing data ---
 --------------------------------------------------------------
    dist     max/min: %16.8f %16.8f
    energy   max/min: %16.8f %16.8f
    gradient max/min: %16.8f %16.8f
    nac      max/min: %16.8f %16.8f
""" % (np.amax(invr),np.amin(invr),np.amax(energy),np.amin(energy),np.amax(gradient),np.amin(gradient),np.amax(nac),np.amin(nac))

    invr_train,invr_val,invr_test=partition(gl_seed,invr)
    energy_train,energy_val,energy_test=partition(gl_seed,energy)
    gradient_train,gradient_val,gradient_test=partition(gl_seed,gradient)
    nac_train,nac_val,nac_test=partition(gl_seed,nac)

    data_info+="""
                  --- Prepare data ---
 --------------------------------------------------------------
    seed: %8d
    dist     train/validation/test: %5d %5d %5d
    energy   train/validation/test: %5d %5d %5d
    gradient train/validation/test: %5d %5d %5d
    nac      train/validation/test: %5d %5d %5d
""" % (gl_seed,len(invr_train),len(invr_val),len(invr_test),len(energy_train),len(energy_val),len(energy_test),len(gradient_train),len(gradient_val),len(gradient_test),len(nac_train),len(nac_val),len(nac_test))

    if stat !=0:
        Statistics(invr,energy,gradient,nac)

    log=open('model.log','a')
    log.write(data_info)
    log.close()

    if silent ==0:
        print (data_info)

    postdata={
    'invr'    :invr,    'miu_invr'    :miu_invr,    'sgm_invr'    :sgm_invr,    'invr_train'    :invr_train,    'invr_val'    :invr_val,    'invr_test'    :invr_test,
    'energy'  :energy,  'miu_energy'  :miu_energy,  'sgm_energy'  :sgm_energy,  'energy_train'  :energy_train,  'energy_val'  :energy_val,  'energy_test'  :energy_test,
    'gradient':gradient,'miu_gradient':miu_gradient,'sgm_gradient':sgm_gradient,'gradient_train':gradient_train,'gradient_val':gradient_val,'gradient_test':gradient_test,
    'nac'     :nac,     'miu_nac'     :miu_nac,     'sgm_nac'     :sgm_nac,     'nac_train'     :nac_train,     'nac_val'     :nac_val,     'nac_test'     :nac_test,
    }
    return postdata

#    return train_invr,val_invr,test_invr,avg_invr,train_energy,val_energy,test_energy,avg_energy,train_gradient,val_gradient,test_gradient,avg_gradient,train_nac,val_nac,test_nac,avg_nac

def NNEG(feat_train,target_train,feat_val,target_val,params):
    ep=params['ep']
    bs=params['bs']
    nlayer=params['nlayer']
    node=params['node']
    wl2=params['wl2']
    lr=params['lr']
    flr=params['flr']
    flrstep=params['flrstep']
    in_weight=params['in_weight']
    silent=params['silent']

    e_train,g_train=target_train
    e_val,g_val=target_val

    dim_in=len(feat_train[0])
    dim_out_e=len(e_train[0])
    dim_out_g=len(g_train[0])

    ## input layer
    input=Input(shape=(dim_in,))
    dense_e=Dense(node,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(input)
    dense_e=BatchNormalization()(dense_e)
    dense_g=Dense(node,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(input)
    dense_g=BatchNormalization()(dense_g)

    ## hidden layers
    for hd in range(nlayer):
        dense_e=Dense(node,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(dense_e)
        dense_e=BatchNormalization()(dense_e)
        dense_g=Dense(node,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(dense_g)
        dense_g=BatchNormalization()(dense_g)

    ## output layer
    dense_e=Dense(dim_out_e,kernel_regularizer=regularizers.l2(wl2),activation='linear',name='e')(dense_e)
    dense_g=Dense(dim_out_g,kernel_regularizer=regularizers.l2(wl2),activation='linear',name='g')(dense_g)

    model=Model(inputs=input,outputs=[dense_e,dense_g])
    model.name="double"
    target_train_dict={'e':e_train,'g':g_train}
    target_val_dict={'e':e_val,'g':g_val}
    adam = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
    optimizer=adam,
    loss={'e':'mean_squared_error','g':'mean_squared_error'},
    loss_weights={'e':0.5,'g':0.5},
    metrics={'e':['mae',rmse,dmax],'g':['mae',rmse,dmax]}
    )

    if silent == 0:
        print(model.summary())

    if in_weight == -1:
        model.load_weights('trained-eg.h5')
        history = model.predict(
        feat_train
        )
    else:
        if in_weight >0:
            model.load_weights('model-eg-%d.h5' % (in_weight))
        history = model.fit(
        feat_train,
        target_train,
        epochs=ep,
        batch_size=bs,
        callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)],
        validation_data=[feat_val,target_val],
        shuffle=True
        )

    return history, model

def NN(feat_train,target_train,feat_val,target_val,params):
    ep=params['ep']
    bs=params['bs']
    nlayer=params['nlayer']
    node=params['node']
    wl2=params['wl2']
    lr=params['lr']
    flr=params['flr']
    flrstep=params['flrstep']
    in_weight=params['in_weight']
    silent=params['silent']

    dim_in=len(feat_train[0])
    dim_out=len(target_train[0])
    ## input layer
    model = Sequential([
      Dense(node, input_shape=(dim_in,),kernel_regularizer=regularizers.l2(wl2),activation='tanh'),
      BatchNormalization()
    ])
    model.name="single"
    ## hidden layers
    for hd in range(nlayer):
        model.add(Dense(node,kernel_regularizer=regularizers.l2(wl2),activation='tanh'))
        model.add(BatchNormalization())
    ## output layer
    model.add(Dense(dim_out,kernel_regularizer=regularizers.l2(wl2),activation='linear'))

    adam = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
    optimizer=adam,
    loss='mean_squared_error',
    metrics=['mae',rmse,dmax],
    )

    if silent == 0:
        print(model.summary())

    if in_weight == -1:
        model.load_weights('trained-eg.h5')
        history = model.predict(
        feat_train
        )
    else:
        if in_weight >0:
            model.load_weights('model-eg-%d.h5' % (in_weight))
        history = model.fit(
        feat_train,
        target_train,
        epochs=ep,
        batch_size=bs,
        callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)],
        validation_data=[feat_val,target_val],
        shuffle=True
        )

    return history, model


def Main(usage,options):
    start=time.time()
    topline='Start: %20s\n%s' % (whatistime(),usage)
    log=open('model.log','w')
    log.write(topline)
    log.close()
    fin=open('model.sum','w')
    fin.write('')
    fin.close()

    global flr,flrstep
    in_data=options.in_data
    pred_data=options.pred_data
    silent=options.silent
    gl_seed=options.gl_seed
    in_weight=options.in_weight
    stat=options.stat
    model_name=options.model_name
    ep=options.ep
    bs=options.bs
    nlayer=options.nlayer
    node=options.node
    wl2=options.wl2
    lr=options.lr
    flr=options.flr
    flrstep=options.flrstep
    nsample=options.nsample
    space_ep=options.s_ep
    space_bs=options.s_bs
    space_nlayer=options.s_nlayer
    space_node=options.s_node
    space_wl2=options.s_wl2
    space_lr=options.s_lr
    space_flr=options.s_flr
    space_flrstep=options.s_flrstep
    s_iter=options.s_iter
    s_win=options.s_win

    if silent == 0:
        print(topline)

    params={
    'ep':ep,
    'bs':bs,
    'nlayer':nlayer,
    'node':node,
    'wl2':wl2,
    'lr':lr,
    'flr':flr,
    'flrstep':flrstep,
    'in_weight':in_weight,
    'silent':silent
    }

    with open('%s' % in_data,'r') as indata:
        data=json.load(indata)
    postdata=Prepdata(data,silent,gl_seed,stat)
    invr_train=postdata['invr_train']
    invr_val=postdata['invr_val']
    energy_train=postdata['energy_train']
    energy_val=postdata['energy_val']
    gradient_train=postdata['gradient_train']
    gradient_val=postdata['gradient_val']
    nac_train=postdata['nac_train']
    nac_val=postdata['nac_val']
    eg_train=[postdata['energy_train'],postdata['gradient_train']]
    eg_val=[postdata['energy_val'],postdata['gradient_val']]

    run_info="""
                  --- Run mode %d
 --------------------------------------------------------------
   -2 Hyperparameter search |   0 New train
   -1 Prediction            |  >0 Load weights
""" % (in_weight)

    if silent == 0:
        print(run_info)

    if   in_weight >= 0:  # New train or restart train
        if   model_name == 'eg':
            history,model=NNEG(invr_train,eg_train,invr_val,eg_val,params)
        elif model_name == 'e':
            history,model=NN(invr_train,energy_train,invr_val,energy_val,params)
        elif model_name == 'g':
            history,model=NN(invr_train,gradient_train,invr_val,gradient_val,params)
        elif model_name == 'nac':
            history,model=NN(invr_train,nac_train,invr_val,nac_val,params)
        train_info="""
                  --- Start training ---
 --------------------------------------------------------------
    model: %20s
    epock: %20d
    batch: %20d
    layer: %20d
     node: %20d
     rate: %20.16f
    decay: %20.16f
    L2reg: %20.16f

                  --- Model summary ---
 --------------------------------------------------------------
"""% (model_name,ep,bs,nlayer,node,lr,flr,wl2)

        log=open('model.log','a')
        log.write(train_info)
        log.close()

        model.save_weights('model-%s-%d.h5' % (model_name,in_weight+1))
        hist=history.history
        arch=[]
        model.summary(print_fn=lambda x: arch.append(x))
        arch= '\n'.join(arch)
        Record(model_name,arch,hist)

    elif in_weight == -1: # Prediction
        with open('%s' % pred_data,'r') as preddata:
            pred=json.load(preddata)
        miu_invr=postdata['miu_invr']
        sgm_invr=postdata['sgm_invr']
        miu_energy=postdata['miu_energy']
        sgm_energy=postdata['sgm_energy']
        miu_gradient=postdata['miu_gradient']
        sgm_gradient=postdata['sgm_gradient']
        miu_nac=postdata['miu_nac']
        sgm_nac=postdata['sgm_nac']

        pred_natom,pred_nstate,pred_invr,pred_energy,pred_gradient,pred_nac=pred
        invr_train=(pred_invr-miu_invr)/sgm_invr
        energy_train=(pred_energy-miu_energy)/sgm_energy
        gradient_train=(pred_gradient-miu_gradient)/sgm_gradient
        nac_train=(pred_nac-miu_nac)/sgm_nac

        if   model_name == 'eg':
            history,model=NNEG(invr_train,eg_train,invr_val,eg_val,params)
        elif model_name == 'e':
            history,model=NN(invr_train,energy_train,invr_val,energy_val,params)
        elif model_name == 'g':
            history,model=NN(invr_train,gradient_train,invr_val,gradient_val,params)
        elif model_name == 'nac':
            history,model=NN(invr_train,nac_train,invr_val,nac_val,params)

    elif in_weight == -2: # Hyperparameter search
        p={
        'ep':params_space(space_ep,ep,'ep'),
        'bs':params_space(space_bs,bs,'bs'),
        'nlayer':params_space(space_nlayer,nlayer,'nlayer'),
        'node':params_space(space_node,node,'node'),
        'wl2':params_space(space_wl2,wl2,'wl2'),
        'lr':params_space(space_lr,lr,'lr'),
        'flr':params_space(space_flr,flr,'flr'),
        'flrstep':params_space(space_flrstep,flrstep,'flrstep'),
        'in_weight':[in_weight],
        'silent':[silent]
        }

        for i in range(s_iter):
            Permut=len(p['ep'])*len(p['bs'])*len(p['nlayer'])*len(p['node'])*len(p['lr'])*len(p['lr'])*len(p['flrstep'])*len(p['wl2'])

            if   int(Permut) >= s_win and int(nsample*Permut) <= s_win:
                nsample=float(s_win)/Permut  # increase sample ratio
            elif int(Permut) <  s_win:
                nsample=1                    # search all space

            search_info="""
                  --- Search space %d ---
 --------------------------------------------------------------
    epock: %50s
    batch: %50s
    layer: %50s
     node: %50s
     rate: %50s
    decay: %50s
     wait: %50s
    L2reg: %50s
    Ratio: %50s
    Total: %50s
   Sample: %50s
   Window: %50s
""" % (i+1,p['ep'],p['bs'],p['nlayer'],p['node'],p['lr'],p['lr'],p['flrstep'],p['wl2'],nsample,Permut,int(nsample*Permut),s_win)
            log=open('model.log','a')
            log.write(search_info)
            log.close()

            if silent == 0:
                print(search_info)

            if int(nsample*Permut) == 1:
                break                        # not enough space to search

            if   model_name == 'eg':
                h = ta.Scan(x=invr_train,y=eg_train,x_val=invr_val,y_val=eg_val,
                model=NNEG,params=p,experiment_name=model_name,
                #reduction_method='', reduction_metric='val_mae',
                random_method='quantum',seed=gl_seed,fraction_limit=nsample)

            elif model_name == 'e':
                history,model=NN(invr_train,energy_train,invr_val,energy_val,params)
            elif model_name == 'g':
                history,model=NN(invr_train,gradient_train,invr_val,gradient_val,params)
            elif model_name == 'nac':
                history,model=NN(invr_train,nac_train,invr_val,nac_val,params)

            candidates=np.array(h.data)
            candidates=candidates[np.argsort(candidates[:,1])]   # sort as val_loss
            candidates=candidates[0:s_win]                       # select candidates
            if i == 0:
                candidates_group=np.copy(candidates) #generate a group of candidates
            else:
                candidates_group=np.concatenate((candidates_group,candidates))  # add candidates to group
                candidates=candidates_group[np.argsort(candidates_group[:,1])]  # sort group as val_loss
                candidates=candidates[0:s_win]                                  # select candidates

            search_info="""
                  --- Search results %d ---
 --------------------------------------------------------------
  Candidates: %6d Group: %6d
  %5s%6s%6s%6s%6s%20s%20s%6s%20s%16s
""" % (i+1,len(candidates),len(candidates_group),'No.','epock','batch','layer','node','rate','decay','wait','L2reg','val_loss')
            for j in range(s_win):
                #index={'bs':-10,'ep':-9,'flr':-8,'flrstep':-7,'lr':-5,'nlayer':-4,'node':-3,'wl2':-1}
                search_info+='  %5d%6d%6d%6d%6d%20.16f%20.16f%6d%20.16f%16.8f\n' % (j+1,candidates[j][-9],candidates[j][-10],candidates[j][-4],candidates[j][-3],candidates[j][-5],candidates[j][-8],candidates[j][-7],candidates[j][-1],candidates[j][1])

            search_info+='\n'
            log=open('model.log','a')
            log.write(search_info)
            log.close()

            if silent == 0:
                print(h.details)
                print(search_info)

            space_ep=update_params(candidates,space_ep,'ep')
            space_bs=update_params(candidates,space_bs,'bs')
            space_nlayer=update_params(candidates,space_nlayer,'nlayer')
            space_node=update_params(candidates,space_node,'node')
            space_wl2=update_params(candidates,space_wl2,'wl2')
            space_lr=update_params(candidates,space_lr,'lr')
            space_flr=update_params(candidates,space_flr,'flr')
            space_flrstep=update_params(candidates,space_flrstep,'flrstep')

            p={
            'ep':params_space(space_ep,ep,'ep'),
            'bs':params_space(space_bs,bs,'bs'),
            'nlayer':params_space(space_nlayer,nlayer,'nlayer'),
            'node':params_space(space_node,node,'node'),
            'wl2':params_space(space_wl2,wl2,'wl2'),
            'lr':params_space(space_lr,lr,'lr'),
            'flr':params_space(space_flr,flr,'flr'),
            'flrstep':params_space(space_flrstep,flrstep,'flrstep'),
            'in_weight':[in_weight],
            'silent':[silent]
            }

    end=time.time()
    walltime=howlong(start,end)
    endline='End: %20s Total: %20s\n' % (whatistime(),walltime)

    log=open('model.log','a')
    log.write(endline)
    log.close()

    usage='%s\n' % (end-start)

    fin=open('model.sum','a')
    fin.write(usage)
    fin.close()

    if silent == 0:
        print('\n%s' % endline)

Main(usage,options)
