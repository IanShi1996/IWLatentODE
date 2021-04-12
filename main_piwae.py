import sys, os
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./notebooks'))

from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device('cuda:0')

# Custom lib
from utils import gpu
from model import LatentNeuralODEBuilder
from train_piw import PIWTrainingLoop

parser = ArgumentParser()

parser.add_argument('--data', type=str, required=True)
parser.add_argument('--model', type=str, choices=['piwae'],
                    required=True)
parser.add_argument('--M', type=int, required=True)
parser.add_argument('--K', type=int, required=True)
parser.add_argument('--ckpt_int', type=int, required=False)
args = parser.parse_args()

data_dict = {
    "sine": "./data/sine_data_2021-04-09 00:13:41.249505",
    "aussign": "./data/aussign_parsed",
}

data_path = data_dict[args.data]

if args.data == "sine":
    generator = torch.load(data_path)['generator']

    train_time, train_data = generator.get_train_set()
    val_time, val_data = generator.get_val_set()

    train_data = train_data.reshape(len(train_data), -1, 1)
    val_data = val_data.reshape(len(val_data), -1, 1)

elif args.data == "aussign":
    data = torch.load(data_path)

    train_data = data["train_dataset"]
    val_data = data["val_dataset"]

    train_time = list(range(train_data.shape[1]))
    val_time = list(range(val_data.shape[1]))
else:
    raise ValueError("Invalid Dataset.")


train_data_tt = gpu(train_data)
train_time_tt = gpu(train_time)

val_data_tt = gpu(val_data)
val_time_tt = gpu(val_time)


class GenericSet(Dataset):
    def __init__(self, data, time):
        self.data = data
        self.time = time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.time


train_dataset = GenericSet(train_data_tt, train_time_tt)
val_dataset = GenericSet(val_data_tt, val_time_tt)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=len(val_dataset))

model_args = {
    'obs_dim': 22,
    'rec_latent_dim': 50,
    'node_latent_dim': 25,

    'rec_gru_unit': 150,
    'rec_node_hidden': 150,
    'rec_node_layer': 2,
    'rec_node_act': 'Tanh',

    'latent_node_hidden': 150,
    'latent_node_layer': 2,
    'latent_node_act': 'Tanh',

    'dec_type': 'NN',
    'dec_hidden': 100,
    'dec_layer': 1,
    'dec_act': 'ReLU',
}

base_model = LatentNeuralODEBuilder(**model_args)
model = base_model.build_latent_node(args.model).to(device)

main = PIWTrainingLoop(model, train_loader, val_loader)

lr = 5e-3

inf_params = (model.enc.parameters())
inf_optim = optim.Adamax(inf_params, lr=lr)
inf_sched = ExponentialLR(inf_optim, 0.9925)

node_params = list(model.nodef.parameters())
dec_params = list(model.dec.parameters())
gen_params = node_params + dec_params

gen_optim = optim.Adamax(gen_params, lr=lr)
gen_sched = ExponentialLR(gen_optim, 0.99)

train_args = {
    'max_epoch': 750,
    'l_std': 0.1,
    'clip_norm': 5,
    'model_atol': 1e-4,
    'model_rtol': 1e-3,
    'plt_args': {'n_plot': 2},
    'M': args.M,
    'K': args.K,
    'method': 'dopri5',
}

out_path = './models/{}_{}_lode_{}_{}'.format(args.data, args.model, args.M, args.K)

if args.ckpt_int:
    train_args['ckpt_int'] = args.ckpt_int

main.train(inf_optim, gen_optim, train_args, inf_sched, gen_sched,
           plt_traj=False, plt_loss=False)

torch.save({
    'model_state_dict': model.state_dict(),
    'inf_opt_state_dict': inf_optim.state_dict(),
    'gen_opt_state_dict': gen_optim.state_dict(),
    'data_path': data_path,
    'model_args': model_args,
    'train_args': train_args,
    'train_obj': main,
}, '{}_{}'.format(out_path, datetime.now()))
