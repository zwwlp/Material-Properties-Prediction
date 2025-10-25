#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import mobopt as mo
import deap.benchmarks as db
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from models1 import GGNN
from model_utils import Model
from data_utils import GraphDataset, graph_collate
import os
import json
import numpy as np
import warnings
import random
from collections import Counter

import random
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def use_setpLR(param):
    ms = param["milestones"]
    return ms[0] < 0

def use_ExponentialLR_gamma(param):
    ms=param["ExponentialLR_gamma"]
    return ms



def count_params(model, dtype=torch.float32) -> float:
    params = sum(p.numel() for p in model.parameters())  # ×Ü²ÎÊýÁ¿
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()  # µ¥²ÎÊý×Ö½ÚÊý
    size_mb = (params * bytes_per_param) / (1024 ** 2)  # ×ª»»ÎªMB
    return size_mb

def create_model(device, model_param, optimizer_param, scheduler_param):
    model = GGNN(**model_param).to(device)
    clip_value = optimizer_param.pop("clip_value")
    optim_name = optimizer_param.pop("optim")
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), momentum=0.9,
                              nesterov=True, **optimizer_param)
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_param)
    elif optim_name == "amsgrad":
        optimizer = optim.Adam(model.parameters(), amsgrad=True,
                               **optimizer_param)
    else:
        raise NameError("optimizer {} is not supported".format(optim_name))
    use_cosine_annealing = scheduler_param.pop("cosine_annealing")
    use_ExponentialLR = scheduler_param.pop("ExponentialLR")
    if use_cosine_annealing:
        params = dict(T_max=scheduler_param["milestones"][0],
                      eta_min=scheduler_param["gamma"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif use_setpLR(scheduler_param):
        scheduler_param["step_size"] = abs(scheduler_param["milestones"][0])
        scheduler_param.pop("milestones")
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
    elif use_ExponentialLR:
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=use_ExponentialLR_gamma(scheduler_param))
    else:
        scheduler_param.pop("ExponentialLR_gamma")
        scheduler_param.pop("gamma")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
    return Model(device, model, optimizer, scheduler, clip_value)

def experiment(options,device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         num_epochs, seed, load_model):
    print("Seed:", seed)
    print()
    torch.manual_seed(seed)
    seed_value = 1234  # Éè¶¨Ëæ»úÊýÖÖ×Ó

    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    dataloader_param["collate_fn"] = graph_collate

    # Create dataset
    #dataset = GraphDataset(options["dataset_path"], options["target_name"], options["atom_feature_onehot"])

    # split the dataset into training, validation, and test sets.
    split_file_path = dataset_param["split_file"]
    if split_file_path is not None and os.path.isfile(split_file_path):
        with open(split_file_path) as f:
            split = json.load(f)
    else:
        print("No split file. Default split: 256 (train), 32 (val), 32 (test)")
        split = {"train": range(256), "val": range(256, 288), "test": range(288, 320)}
    print(" ".join(["{}: {}".format(k, int(options["sampler_size"]*len(x))) for k, x in split.items()]))

    # Create a CGNN model
    model = create_model(device, model_param, optimizer_param, scheduler_param)
    model_param_size=count_params(model.model)
    # print('Total params: %.2fM' % (model_param_size))
    if load_model:
        print("Loading weights from model.pth")
        model.load()
    #print("Model:", model.device)
    dataset = GraphDataset(options["dataset_path"], options["target_name"], options["atom_feature_onehot"])
    # Train
    train_sampler = SubsetRandomSampler(np.random.choice(split["train"], int(options["sampler_size"]*len(split["train"])),replace=False))
    val_sampler = SubsetRandomSampler(np.random.choice(split["val"], int(options["sampler_size"]*len(split["val"])),replace=False))
    train_dl = DataLoader(dataset, sampler=train_sampler, **dataloader_param)
    val_dl = DataLoader(dataset, sampler=val_sampler, **dataloader_param)
    model.train(train_dl, val_dl, num_epochs)
    if num_epochs > 0:
        model.save()

    # Test
    test_set = Subset(dataset, split["test"])
    test_dl = DataLoader(test_set, **dataloader_param)
    outputs, targets,total_metrics= model.evaluate(test_dl)
    loss=total_metrics[1][1]
    #names = [dataset.graph_names[i] for i in split["test"]]
    #df_predictions = pd.DataFrame({"name": names, "prediction": outputs, "target": targets})
    #FileName = "lr_{:06f}_gamma{:03f}_name{:}". \
                   #format(options["lr"], options["ExponentialLR_gamma"], options["target_name"]) + '.csv'
    #df_predictions.to_csv('../save_data/'+FileName, index=False)
    print("\nEND")
    return [model_param_size, loss]

def target(x):
    import argparse
    feature_num = Counter(x > 1.9)[True]
    parser = argparse.ArgumentParser(description="Crystal Graph Neural Networks")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_node_feat", type=int, default=89)
    parser.add_argument("--n_hidden_feat", type=int, default=64)
    parser.add_argument("--n_graph_feat", type=int, default=128)
    parser.add_argument("--n_nbr_fea",type=int,default=44)
    parser.add_argument("--n_conv", type=int, default=3)
    parser.add_argument("--n_fc", type=int, default=2)
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--use_batch_norm", action='store_true')
    parser.add_argument("--node_activation", type=str, default="None")
    parser.add_argument("--use_node_batch_norm", action='store_true')
    parser.add_argument("--edge_activation", type=str, default="None")
    parser.add_argument("--use_edge_batch_norm", action='store_true')
    parser.add_argument("--n_edge_net_feat", type=int, default=32)#32)
    parser.add_argument("--n_edge_net_layers", type=int, default=3)
    parser.add_argument("--edge_net_activation", type=str, default="elu")
    parser.add_argument("--use_edge_net_batch_norm", action='store_true')
    parser.add_argument("--use_fast_edge_network", action='store_true')
    parser.add_argument("--dropout",type=int,default=0.6)
    parser.add_argument("--fast_edge_network_type", type=int, default=0)
    parser.add_argument("--use_aggregated_edge_network", action='store_true')
    parser.add_argument("--edge_net_cardinality", type=int, default=32)
    parser.add_argument("--edge_net_width", type=int, default=4)
    parser.add_argument("--use_edge_net_shortcut", action='store_true')
    parser.add_argument("--use_attention", action='store_true',default=False)
    parser.add_argument("--n_postconv_net_layers", type=int, default=0)
    parser.add_argument("--postconv_net_activation", type=str, default="elu")
    parser.add_argument("--use_postconv_net_batch_norm", action='store_true')
    parser.add_argument("--use_distance_nbr",action='store_true',default=True)
    parser.add_argument("--conv_bias", action='store_true')
    parser.add_argument("--edge_net_bias", action='store_true')
    parser.add_argument("--postconv_net_bias", action='store_true')
    parser.add_argument("--full_pooling", action='store_true')
    parser.add_argument("--gated_pooling", action='store_true')
    parser.add_argument("--conv_type", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--milestones", nargs='+', type=int, default=[10])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--cosine_annealing", action='store_true')
    parser.add_argument("--ExponentialLR", action='store_true',default=True)
    parser.add_argument("--ExponentialLR_gamma",default=0.85)
    parser.add_argument("--lr_decay_steps",type=str,default=[700,1000])
    parser.add_argument("--num_epochs", type=int, default=550)
    parser.add_argument("--dataset_path", type=str, default="/home/zww/program/cgnn-attention/OQMD")
    parser.add_argument("--target_name", type=str, default="formation_energy_per_atom")
    parser.add_argument("--split_file", type=str, default="/home/zww/program/cgnn-attention/OQMD/split_cutoff.json")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--load_model", action='store_true',default=False)
    parser.add_argument("--use_extension", action='store_true')
    parser.add_argument("--atom_feature_onehot",action='store_true',default=True)
    if feature_num > 0 :
        parser.add_argument("--feature_encoding",default=x > 1.5)
    else:
        parser.add_argument("--feature_encoding",default=list(map(bool,[1]+[0]*12)))

    parser.add_argument("--sampler_size",type=float,default=0.2)
    options = vars(parser.parse_args())

    if not torch.cuda.is_available():
        options["device"] = "cpu"
    print("Device:", options["device"])
    print()
    device = torch.device(options["device"])

    # Model parameters
    model_param_names = [
        "n_node_feat", "n_hidden_feat", "n_graph_feat","n_nbr_fea", "n_conv", "n_fc",
        "activation", "use_batch_norm", "node_activation", "use_node_batch_norm",
        "edge_activation", "use_edge_batch_norm", "n_edge_net_feat","n_edge_net_layers",
        "edge_net_activation", "use_edge_net_batch_norm","use_fast_edge_network","dropout",
        "fast_edge_network_type","use_aggregated_edge_network", "use_distance_nbr","edge_net_cardinality",
        "edge_net_width", "use_edge_net_shortcut", "n_postconv_net_layers",
        "postconv_net_activation","use_attention","use_postconv_net_batch_norm", "conv_type",
        "conv_bias", "edge_net_bias", "postconv_net_bias","feature_encoding",
        "full_pooling", "gated_pooling", "use_extension",]
    model_param = {k : options[k] for k in model_param_names if options[k] is not None}
    if model_param["node_activation"].lower() == 'none':
        model_param["node_activation"] = None
    if model_param["edge_activation"].lower() == 'none':
        model_param["edge_activation"] = None
    print("Model:", model_param)
    print()

    # Optimizer parameters
    optimizer_param_names = ["optim", "lr", "weight_decay", "clip_value"]
    optimizer_param = {k : options[k] for k in optimizer_param_names if options[k] is not None}
    if optimizer_param["clip_value"] == 0.0:
        optimizer_param["clip_value"] = None
    print("Optimizer:", optimizer_param)
    print()

    # Scheduler parameters
    scheduler_param_names = ["milestones", "gamma", "cosine_annealing","ExponentialLR","ExponentialLR_gamma"]
    scheduler_param = {k : options[k] for k in scheduler_param_names if options[k] is not None}
    print("Scheduler:", scheduler_param)
    print()

    # Dataset parameters
    dataset_param_names = ["dataset_path", "target_name", "split_file"]
    dataset_param = {k : options[k] for k in dataset_param_names if options[k] is not None}
    print("Dataset:", dataset_param)
    print()

    # Dataloader parameters
    dataloader_param_names = ["num_workers", "batch_size"]
    dataloader_param = {k : options[k] for k in dataloader_param_names if options[k] is not None}
    print("Dataloader:", dataloader_param)
    print()

    model_param_size,loss=experiment(options,device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         options["num_epochs"], options["seed"], options["load_model"])
    #return np.asarray([feature_num,loss])
    if feature_num==4:
        print(model_param_size)
    return np.asarray([model_param_size,loss])
    # if feature_num > 0:
    #     return np.asarray([Counter(x > 1.9)[True], loss])
    # else:
    #     return np.asarray([1, loss])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="the MOBOpt params")
    parser.add_argument("-d", dest="ND", type=int, metavar="ND",
                        help="Number of Dimensions for ZDT1",
                        default=13,
                        required=False)
    parser.add_argument("-i", dest="NI", type=int, metavar="NI",
                        help="Number of iterations of the method",
                        default=55,
                        required=False)
    parser.add_argument("-r", dest="Prob", type=float, default=0.1,
                        help="Probability of random jumps",
                        required=False)
    parser.add_argument("-q", dest="Q", type=float, default=1.0,
                        help="Weight in factor",
                        required=False)
    parser.add_argument("-ni", dest="NInit", type=int, metavar="NInit",
                        help="Number of initialization points",
                        required=False, default=5)
    parser.add_argument("-nr", dest="NRest", type=int, metavar="N Restarts",
                        help="Number of restarts of GP optimizer",
                        required=False, default=100)
    parser.add_argument("-v", dest="verbose", action='store_true',default=True,
                        help="Verbose")
    parser.add_argument("--filename", dest="Filename", type=str,
                        default="cgnn-attention.csv",
                        help="Filename for saving data")
    parser.add_argument("-r-prob", dest="Reduce", action="store_true",
                        help="If present reduces prob linearly" +
                        " along simmulation")
    parser.set_defaults(Reduce=False)

    args = parser.parse_args()

    NParam = args.ND#参数维度
    NIter = args.NI
    if 0 <= args.Prob <= 1.0:
        Prob = args.Prob
    else:
        raise ValueError("Prob must be between 0 and 1")
    N_init = args.NInit
    verbose = args.verbose#if print immediat  information
    Q = args.Q#

    PB = np.asarray([[0, 12]]*NParam)#参数范围
    # PB = np.asarray([[0,1],
    #                  [0,1]])#/100

    Optimize = mo.MOBayesianOpt(target=target,
                                NObj=2,
                                pbounds=PB,
                                Picture=True,
                                MetricsPS=False,
                                verbose=verbose,
                                n_restarts_optimizer=args.NRest,
                                Filename=args.Filename,
                                max_or_min='min')

    Optimize.initialize(init_points=N_init)

    front, pop = Optimize.maximize(n_iter=NIter,
                                   prob=Prob,
                                   q=Q,
                                   SaveInterval=10,
                                   FrontSampling=[80],
                                   ReduceProb=args.Reduce)

    PF = np.asarray([np.asarray(y) for y in Optimize.y_Pareto])
    PS = np.asarray([np.asarray(x) for x in Optimize.x_Pareto])

    FileName = "FF_D{:02d}_I{:04d}_NI{:02d}_P{:4.2f}_Q{:4.2f}".\
               format(NParam, NIter, N_init, Prob, Q) + args.Filename
    # df_predictions = pd.DataFrame({"PF":-PF, "PS":PS})
    # df_predictions = pd.DataFrame({"PF_param_size": -PF[:,0], "PF_loss": -PF[:,1], "PS_n_fea": PS[:,0], "PS_gamma": PS[:,1]})
    # df_predictions = pd.DataFrame(
    #     {"PF_param_size": -PF[:, 0], "PF_loss": -PF[:, 1], "_nPS_fea": PS[:, 0], "PS_gamma": PS[:, 1]})
    #
    # df_predictions.to_csv('../save_data_feature_optimization/'+FileName, index=[0])
    np.savez('../save_data_feature_optimization/'+FileName,
             Front=front,
             Pop=pop,
             PF=-PF,
             PS=PS)

    fig, ax = pl.subplots(1, 1)
    ax.scatter(-PF[:, 0],-PF[:, 1],label="$\chi$")
    #ax.scatter(-PF[:, 0], ((0.03-0.0195)/(max(-PF[:, 1])-min(-PF[:, 1])))*(-PF[:, 1]-min(-PF[:, 1]))+0.0195, label="$\chi$")
    ax.grid()
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.legend()
    fig.savefig('../save_data_feature_optimization/'+FileName+".png", dpi=600)
    #fig.savefig('./save_data_feature_optimization/'+FileName+".eps", dpi=900)

    # GenDist = mo.metrics.GD(front, np.asarray([f1, f2]).T)
    # Delta = mo.metrics.Spread2D(front, np.asarray([f1, f2]).T)

    # if verbose:
    #     print("GenDist = ", GenDist)
    #     print("Delta = ", Delta)
    #
    # pass


if __name__ == '__main__':
    main()
