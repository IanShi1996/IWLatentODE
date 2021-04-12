import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./notebooks'))

from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device('cuda:0')

# Custom lib
from model import LatentNeuralODEBuilder
from train import TrainingLoop
from train_piw import PIWTrainingLoop
from data_utils import get_dataloaders, DATA_PATH_DICT

parser = ArgumentParser()

parser.add_argument('--data', type=str, required=True)
parser.add_argument('--model', type=str, choices=['base', 'iwae', 'miwae',
                                                  'ciwae', 'betavae', 'piwae'],
                    required=True)
parser.add_argument('--M', type=int, required=True)
parser.add_argument('--K', type=int, required=True)
parser.add_argument('--beta', type=float, required=False)
parser.add_argument('--ckpt_int', type=int, required=False)
args = parser.parse_args()

batch_size = 128
train_loader, val_loader = get_dataloaders(args.data, batch_size)

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

lr = 1e-3
decay = 0.995

train_args = {
    'max_epoch': 400,
    'l_std': 0.01,
    'clip_norm': 5,
    'model_atol': 1e-4,
    'model_rtol': 1e-3,
    'M': args.M,
    'K': args.K,
    'method': 'dopri5',
}

out_path = './models/{}_{}_lode_{}_{}'.format(args.data, args.model, args.M,
                                              args.K)

if args.beta:
    train_args['beta'] = args.beta
    out_path += '_{}'.format(args.beta)

if args.ckpt_int:
    train_args['ckpt_int'] = args.ckpt_int

if args.model == 'piwae':
    main = PIWTrainingLoop(model, train_loader, val_loader)

    inf_params = (model.enc.parameters())
    inf_optim = optim.Adamax(inf_params, lr=lr)
    inf_sched = ExponentialLR(inf_optim, decay)

    node_params = list(model.nodef.parameters())
    dec_params = list(model.dec.parameters())
    gen_params = node_params + dec_params

    gen_optim = optim.Adamax(gen_params, lr=lr)
    gen_sched = ExponentialLR(gen_optim, decay)

    main.train(inf_optim, gen_optim, train_args, inf_sched, gen_sched)

    torch.save({
        'model_state_dict': model.state_dict(),
        'inf_opt_state_dict': inf_optim.state_dict(),
        'gen_opt_state_dict': gen_optim.state_dict(),
        'data_path': DATA_PATH_DICT[args.data],
        'model_args': model_args,
        'train_args': train_args,
        'train_obj': main,
    }, '{}_{}'.format(out_path, datetime.now()))

else:
    main = TrainingLoop(model, train_loader, val_loader)

    parameters = (model.parameters())
    optimizer = optim.Adamax(parameters, lr=lr)
    scheduler = ExponentialLR(optimizer, decay)

    main.train(optimizer, train_args, scheduler)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'data_path': DATA_PATH_DICT[args.data],
        'model_args': model_args,
        'train_args': train_args,
        'train_obj': main,
    }, '{}_{}'.format(out_path, datetime.now()))
