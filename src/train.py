import time

import torch
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

from utils import RunningAverageMeter


class TrainingLoop:
    def __init__(self, model, train_loader, val_loader, plot_func=None,
                 loss_meters=None, loss_hists=None):
        """Initialize main training loop for Neural ODE model.

        Dataloaders should return tuple of data and timepoints with shapes:
            ((B x L x D), (B x L)) where
            B = Batch size, L = Number of observations, D = Data dimension.

        Args:
            model (nn.Module): Model to train.
            train_loader (torch.utils.data.Dataloader): Training data loader.
            val_loader (torch.utils.data.Dataloader): Validation data loader.
            plot_func (function): Function used to plot predictions.
            loss_meters (RunningAverageMeter, RunningAverageMeter):
                Existing training / val loss average meters.
            loss_hists (list, list): Train/val loss history arrays.
        """

        self.model = model
        self.init_loss_history(loss_hists)
        self.init_loss_meter(loss_meters)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.plot_func = plot_func
        self.execution_arg_history = []

        self.runtimes = []

    def init_loss_history(self, loss_hist):
        if loss_hist is None:
            self.train_loss_hist = []
            self.val_loss_hist = []
        else:
            self.train_loss_hist = loss_hist[0]
            self.val_loss_hist = loss_hist[1]

    def init_loss_meter(self, loss_meters):
        if loss_meters is None:
            self.train_loss_meter = RunningAverageMeter()
            self.val_loss_meter = RunningAverageMeter()
        else:
            self.train_loss_meter = loss_meters[0]
            self.val_loss_meter = loss_meters[1]

    def train(self, optimizer, args, scheduler=None, verbose=True,
              plt_traj=False, plt_loss=False):
        """Execute main training loop for Neural ODE model.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer.
            args (dict): Additional training arguments. See below.
            scheduler (torch._LRScheduler): Learning rate scheduler.
            verbose (bool): Prints verbose out training information.
            plt_traj (bool): Plot reconstructions.
            plt_loss (bool): Plot loss history.

        Additional Args:
            args['max_epochs'] (int): Maximum training epochs.
            args['l_std'] (float): Std used to calculate likelihood in ELBO.
            args['clip_norm] (float): Max norm used to clip gradients.
            args['model_atol'] (float): Absolute tolerance used by ODE solve.
            args['model_rtol'] (float): Relative tolerance used by ODE solve.
            args['plt_args'] (dict): Plotting arguments.
        """
        self.execution_arg_history.append(args)

        atol = args['model_atol'] or 1e-5
        rtol = args['model_rtol'] or 1e-7

        epoch_times = []
        for epoch in range(1, args['max_epoch']+1):
            start_time = time.time()

            for b_data, b_time in self.train_loader:
                optimizer.zero_grad()

                out = self.model.forward(b_data, b_time[0], args['M'],
                                         args['K'], rtol, atol, args['method'])
                elbo = self.model.get_elbo(b_data, *out, args['M'], args['K'],
                                           args['l_std'])

                self.train_loss_meter.update(elbo.item())

                elbo.backward()
                if args['clip_norm']:
                    clip_grad_norm_(self.model.parameters(), args['clip_norm'])
                optimizer.step()
            if scheduler:
                scheduler.step()

            end_time = time.time()
            epoch_times.append(end_time - start_time)

            with torch.no_grad():
                self.update_val_loss(args)

                if self.plot_func and plt_traj:
                    self.plot_val_traj(args['plt_args'])
                self.train_loss_hist.append(self.train_loss_meter.avg)
                self.val_loss_hist.append(self.val_loss_meter.val)

            if verbose:
                if scheduler:
                    print("Current LR: {}".format(scheduler.get_last_lr()),
                          flush=True)
                if plt_loss:
                    self.plot_loss()
                self.print_loss(epoch)

        self.runtimes.append(epoch_times)

    def update_val_loss(self, args):
        atol = args['model_atol'] or 1e-5
        rtol = args['model_rtol'] or 1e-7

        val_data_tt, val_tp_tt = next(iter(self.val_loader))
        val_out = self.model.forward(val_data_tt, val_tp_tt[0], args['M'],
                                     args['K'], rtol, atol, args['method'])
        val_elbo = self.model.get_elbo(val_data_tt, *val_out, args['l_std'])

        self.val_loss_meter.update(val_elbo.item())

    def print_loss(self, epoch):
        print('Epoch: {}, Train ELBO: {:.3f}, Val ELBO: {:.3f}'.format(
            epoch, -self.train_loss_meter.avg, -self.val_loss_meter.avg),
            flush=True)

    def plot_loss(self):
        train_range = range(len(self.train_loss_hist))
        val_range = range(len(self.val_loss_hist))
        plt.plot(train_range, self.train_loss_hist, label='train')
        plt.plot(val_range, self.val_loss_hist, label='validation')
        plt.legend()
        plt.show()

    def plot_val_traj(self, args):
        val_data_tt, val_tp_tt = next(iter(self.val_loader))
        self.plot_func(self.model, val_data_tt, val_tp_tt, **args)
