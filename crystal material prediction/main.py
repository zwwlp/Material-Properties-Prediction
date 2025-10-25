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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def use_setpLR(param):
    ms = param["milestones"]
    return ms[0] < 0

def use_ExponentialLR_gamma(param):
    ms=param["ExponentialLR_gamma"]
    return ms


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The network has {} params.'.format(params))
    return params

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

def main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         num_epochs, seed, load_model):
    print("Seed:", seed)
    print()
    torch.manual_seed(seed)

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
    print(" ".join(["{}: {}".format(k, len(x)) for k, x in split.items()]))

    # Create a CGNN model
    model = create_model(device, model_param, optimizer_param, scheduler_param)
    model_param_size=count_params(model.model) / 1000000.0
    print('Total params: %.2fM' % (model_param_size))
    if load_model:
        print("Loading weights from model.pth")
        model.load()
    #print("Model:", model.device)
    dataset = GraphDataset(options["dataset_path"], options["target_name"], options["atom_feature_onehot"])
    # Train
    train_sampler = SubsetRandomSampler(split["train"])
    val_sampler = SubsetRandomSampler(split["val"])
    train_dl = DataLoader(dataset, sampler=train_sampler, **dataloader_param)
    val_dl = DataLoader(dataset, sampler=val_sampler, **dataloader_param)
    model.train(train_dl, val_dl, num_epochs)
    if num_epochs > 0:
        model.save()

    # Test
    test_set = Subset(dataset, split["test"])
    test_dl = DataLoader(test_set, **dataloader_param)
    outputs, targets,total_metrics= model.evaluate(test_dl)
    loss=total_metrics[0][1]
    names = [dataset.graph_names[i] for i in split["test"]]
    df_predictions = pd.DataFrame({"name": names, "prediction": outputs, "target": targets})
    FileName = "lr_{:06f}_gamma{:03f}_name{:}". \
                   format(options["lr"],options["ExponentialLR_gamma"],options["target_name"])+'.csv'
    df_predictions.to_csv(FileName, index=False)
    print("\nEND")
    return [model_param_size, loss]

if __name__ == '__main__':
    import argparse

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
    parser.add_argument("--n_edge_net_feat", type=int, default=32)
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
    parser.add_argument("--use_attention", action='store_true',default=True)
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
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="/home/zww/cgnn-attention/OQMD")
    parser.add_argument("--target_name", type=str, default="formation_energy_per_atom")
    parser.add_argument("--split_file", type=str, default="/home/zww/cgnn-attention/OQMD/split_000.json")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--use_extension", action='store_true')
    parser.add_argument("--atom_feature_onehot",action='store_true',default=True)
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
        "conv_bias", "edge_net_bias", "postconv_net_bias",
        "full_pooling", "gated_pooling", "use_extension"]
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

    [model_param_size,loss]=main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         options["num_epochs"], options["seed"], options["load_model"])

