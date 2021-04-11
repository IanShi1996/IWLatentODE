import time

import torch
from torch.nn.utils import clip_grad_norm_
from train import TrainingLoop, exists_checkpoint, get_checkpoint_path


class PIWTrainingLoop(TrainingLoop):

    def __init__(self, model, train_loader, val_loader, plot_func=None,
                 loss_meters=None, loss_hists=None):
        super().__init__(model, train_loader, val_loader, plot_func,
                         loss_meters, loss_hists)

    def save_checkpoint(self, opt1, opt2, sch1, sch2, epoch, epoch_times,
                        ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = get_checkpoint_path()

        sch1_sd = sch1.state_dict() if sch1 else None
        sch2_sd = sch2.state_dict() if sch2 else None

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'opt1_state_dict': opt1.state_dict(),
            'opt2_state_dict': opt2.state_dict(),
            'sch1_state_dict': sch1_sd,
            'sch2_state_dict': sch2_sd,
            'epoch': epoch,
            'epoch_times': epoch_times,
            'loss_hists': [self.train_loss_hist, self.val_loss_hist],
            'loss_meters': [self.train_loss_meter, self.val_loss_meter],
        }, ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = get_checkpoint_path()

        ckpt = torch.load(ckpt_path)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.init_loss_history(ckpt['loss_hists'])
        self.init_loss_meter(ckpt['loss_meters'])

        epoch_times = ckpt['epoch_times']

        epoch = ckpt['epoch']
        opt1_sd = ckpt['opt1_state_dict']
        opt2_sd = ckpt['opt2_state_dict']
        sch1_sd = ckpt['sch1_state_dict']
        sch2_sd = ckpt['sch2_state_dict']

        return epoch, epoch_times, opt1_sd, opt2_sd, sch1_sd, sch2_sd

    def train(self, opt1, opt2, args, sche1=None, sche2=None, verbose=True,
              plt_traj=False, plt_loss=False):
        """

        Args:
            opt1: Optimizes Encoder Network (inference)
            opt2: Optimizes Decoder Network (generative)

        Returns:

        """
        self.execution_arg_history.append(args)

        start_epoch = 1
        epoch_times = []

        if 'ckpt_int' in args and exists_checkpoint():
            ckpt = self.load_checkpoint()

            start_epoch = ckpt[0]
            epoch_times = ckpt[1]

            opt1.load_state_dict(ckpt[2])
            opt2.load_state_dict(ckpt[3])
            if sche1:
                sche1.load_state_dict(ckpt[4])
            if sche2:
                sche2.load_state_dict(ckpt[5])

        for epoch in range(start_epoch, args['max_epoch']+1):
            start_time = time.time()

            for b_data, b_time in self.train_loader:

                out = self.model.forward(b_data, b_time[0], args)
                iwae_elbo, miwae_elbo = self.model.get_elbo(b_data, *out, args)

                opt1.zero_grad()
                miwae_elbo.backward()

                if 'clip_norm' in args:
                    inf_params = self.model.enc.parameters()
                    clip_grad_norm_(inf_params, args['clip_norm'])
                opt1.step()

                opt2.zero_grad()
                iwae_elbo.backward()

                if 'clip_norm' in args:
                    node_params = list(self.model.nodef.parameters())
                    dec_params = list(self.model.dec.parameters())
                    gen_params = node_params + dec_params

                    clip_grad_norm_(gen_params, args['clip_norm'])
                opt2.step()

                avg_elbo = (iwae_elbo + miwae_elbo) / 2
                self.train_loss_meter.update(avg_elbo.item())

            if sche1:
                sche1.step()

            if sche2:
                sche2.step()

            end_time = time.time()
            epoch_times.append(end_time - start_time)

            with torch.no_grad():
                self.update_val_loss(args)

                if self.plot_func and plt_traj:
                    self.plot_val_traj(args['plt_args'])
                self.train_loss_hist.append(self.train_loss_meter.avg)
                self.val_loss_hist.append(self.val_loss_meter.val)

            if verbose:
                if sche1:
                    print("Current LR: {}".format(sche1.get_last_lr()),
                          flush=True)
                if plt_loss:
                    self.plot_loss()
                self.print_loss(epoch)

            if 'ckpt_int' in args and epoch % args['ckpt_int']:
                self.save_checkpoint(opt1, opt2, sche1, sche2, epoch,
                                     epoch_times)

        self.runtimes.append(epoch_times)
