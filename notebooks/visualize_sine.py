import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import torch

from simulate_sine import generate_sine_linear

sys.path.append(os.path.abspath('../src'))
from utils import gpu, asnp


def visualize_trajectory(data, ts, model, ax=plt.gca()):
    """Visualize trajectory reconstructions by Latent NODE model.

    Given an input of ground truth data, visualizes reconstructed trajectory
    made by latent model. Ground truth trajectories are solid, while
    predictions are dashed.

    Data should be in shape of BxLxD where:
        B = number of trajectories
        L = length of time series
        D = input features

    Args:
        data (np.ndarray): Input data to visualize.
        ts (np.ndarray): Timepoints of observation for data points.
        model (nn.Module): PyTorch model to evaluate.
        ax (matplotlib.axes.Axes): Matplotlib axes to plot results.
    """
    out = asnp(model.get_prediction(gpu(data), gpu(ts)))

    for i in range(len(data)):
        ax.plot(ts, data[i], c='red', alpha=0.8)
        ax.plot(ts, out[i].squeeze(), c='orange', alpha=0.9, linestyle='--')


def plot_training_sine(model, data, tps, n_plot):
    """Plot trajectories in individual subplots.

    Used as callback during training loop.

    Args:
        model (nn.Module): PyTorch model used to get prediction.
        data (torch.Tensor or np.ndarray): Data.
        tps (torch.Tensor or np.ndarray): Timepoints.
        n_plot (int): Number of subplots.
    """
    fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 5))

    ind = np.random.randint(0, len(data), n_plot)

    if isinstance(data, torch.Tensor):
        data = asnp(data)
    data = data[ind]

    if isinstance(tps, torch.Tensor):
        tps = asnp(tps)
    tps = tps[ind]

    for i in range(n_plot):
        d = data[i][np.newaxis, :, :]
        visualize_trajectory(d, tps[i], model, ax=axes[i])
    plt.show()


def visualize_grid(data, pred, ts, titles):
    """Plot predicted data against ground truth in a grid of subplots.

    Plots trajectories reconstructed by the model against the ground truth
    input data. Data is expected to vary across some latent parameter, but
    share observation times. The generated grid will attempt to be as square
    as possible.

    Args:
        data (np.ndarray): Input data. First dimension indexes trajectories.
        pred (np.ndarray): Reconstructed trajectories in same order as data.
        ts (np.ndarray): Data observation time points.
        titles (list of string): Titles of each subplot.
    """
    n_row = int(np.ceil(np.sqrt(len(data))))
    n_col = int(np.ceil(len(data) / n_row))

    fig, ax = plt.subplots(n_row, n_col, sharex=True, sharey=True)
    fig.set_size_inches((n_col * 3, n_row * 2))

    r = 0
    c = 0

    for i in range(len(data)):
        ax[r, c].plot(ts, data[i])
        ax[r, c].plot(ts, pred[i])
        ax[r, c].title.set_text(titles[i])

        ax[r, c].minorticks_on()

        ax[r, c].grid(which='major')
        ax[r, c].grid(which='minor', linestyle='--')

        c += 1
        if c == n_col:
            c = 0
            r += 1

    plt.show()


def visualize_amplitude(n_samp, freq, amps, phase, stop, model):
    """Visualize model performance on different sine wave amplitude settings.

    Args:
        n_samp (int): Number of samples per trajectory.
        freq (float): Frequency of sine waves.
        amps (list of float): Amplitudes to evaluate.
        phase (float): Phase of sine waves.
        stop (float): End timepoint of sine waves.
        model (nn.Module): PyTorch model to evaluate.
    """
    data = np.zeros((len(amps), n_samp))

    for i in range(len(amps)):
        t, d = generate_sine_linear(n_samp, freq, amps[i], phase, stop)
        data[i] = d

    out = asnp(model.get_prediction(gpu(data).unsqueeze(2), gpu(t)))

    titles = ["Amp = {}".format(amp) for amp in amps]

    visualize_grid(data, out, t, titles)


def visualize_frequency(n_samp, freqs, amp, phase, stop, model):
    """Visualize model performance on different sine wave frequency settings.

    Args:
        n_samp (int): Number of samples per trajectory.
        freqs (list of float): Frequencies of sine waves to evaluate.
        amp (float): Amplitude of sine waves.
        phase (float): Phase of sine waves.
        stop (float): End timepoint of sine waves.
        model (nn.Module): PyTorch model to evaluate.
    """
    data = np.zeros((len(freqs), n_samp))

    for i in range(len(freqs)):
        t, d = generate_sine_linear(n_samp, freqs[i], amp, phase, stop)
        data[i] = d

    out = asnp(model.get_prediction(gpu(data).unsqueeze(2), gpu(t)))

    titles = ["Frequency = {}".format(freq) for freq in freqs]

    visualize_grid(data, out, t, titles)


def visualize_time(n_samp, freq, amp, phase, stops, model):
    """Visualize model performance on trajectories of various lengths.

    Args:
        n_samp (int): Number of samples per trajectory.
        freq (float): Frequency of sine waves.
        amp (float): Amplitude of sine waves.
        phase (float): Phase of sine waves.
        stop (list of float): All end timepoint of sine waves to evaluate.
        model (nn.Module): PyTorch model to evaluate.
    """
    data = np.zeros((len(stops), n_samp))

    for i in range(len(stops)):
        t, d = generate_sine_linear(n_samp, freq, amp, phase, stops[i])
        data[i] = d

    out = asnp(model.get_prediction(gpu(data).unsqueeze(2), gpu(t)))

    titles = ["Stop = {}".format(stop) for stop in stops]

    visualize_grid(data, out, t, titles)


def visualize_phase(n_samp, freq, amp, phases, stop, model):
    """Visualize model performance on trajectories of various phases.

    Args:
        n_samps (list of int): All numbers of  trajectory samples to evaluate.
        freq (float): Frequency of sine waves.
        amp (float): Amplitude of sine waves.
        phase (float): Phase of sine waves.
        stop (float): End timepoint of sine waves.
        model (nn.Module): PyTorch model to evaluate.
    """
    data = np.zeros((len(phases), n_samp))

    for i in range(len(phases)):
        t, d = generate_sine_linear(n_samp, freq, amp, phases[i], stop)
        data[i] = d

    out = asnp(model.get_prediction(gpu(data).unsqueeze(2), gpu(t)))

    titles = ["Phase = {}".format(phase) for phase in phases]

    visualize_grid(data, out, t, titles)


def visualize_samples(n_samps, freq, amp, phase, stop, model):
    """Visualize model performance on trajectories of various sample sizes.

    Processing is sequential due to irregularly sampled time series.
    This however makes for very slow runs.

    Args:
        n_samps (list of int): All numbers of  trajectory samples to evaluate.
        freq (float): Frequency of sine waves.
        amp (float): Amplitude of sine waves.
        phase (float): Phase of sine waves.
        stop (float): End timepoint of sine waves.
        model (nn.Module): PyTorch model to evaluate.
    """
    data = []
    ts = []
    pred = []

    # TODO: Optimize by using a padding scheme.
    for i in range(len(n_samps)):
        t, d = generate_sine_linear(n_samps[i], freq, amp, phase, stop)

        out = asnp(model.get_prediction(gpu(d).reshape(1, -1, 1), gpu(t)))

        data.append(d)
        ts.append(t)
        pred.append(out.flatten())

    titles = ["# Samps = {}".format(n_samp) for n_samp in n_samps]

    n_row = int(np.ceil(np.sqrt(len(n_samps))))
    n_col = int(np.ceil(len(n_samps) / n_row))

    fig, ax = plt.subplots(n_row, n_col, sharex=True, sharey=True)
    fig.set_size_inches((n_col * 3, n_row * 2))

    r = 0
    c = 0

    for i in range(len(n_samps)):
        ax[r, c].plot(ts[i], data[i].flatten())
        ax[r, c].plot(ts[i], pred[i])
        ax[r, c].title.set_text(titles[i])

        ax[r, c].minorticks_on()

        ax[r, c].grid(which='major')
        ax[r, c].grid(which='minor', linestyle='--')

        c += 1
        if c == n_col:
            c = 0
            r += 1

    plt.show()


def visualize_segmentation(tp, data, true_cp, pred_cp, model):
    """Visualize segmentation results.

    Plots reconstructed trajectory against input data points. True changepoints
    are visualized against changepoints predicted via PELT.

    TODO: Add feature to show more than one reconstructed trajectory.

    Args:
        tp (np.ndarray): Observation timepoints of data.
        data (np.ndarray): Input data.
        true_cp (np.ndarray): Location of true changepoints.
        pred_cp (np.ndarray): Location of predicted changepoints.
        model (nn.Module): PyTorch Latent NODE model.
    """
    pred_cp = np.concatenate(([0], pred_cp, [len(data)])).astype(int)
    segment = []

    for i in range(len(pred_cp) - 1):
        data_tt = gpu(data[pred_cp[i]:pred_cp[i+1]]).reshape(1, -1, 1)
        tp_tt = gpu(tp[pred_cp[i]:pred_cp[i+1]])

        segment_x = asnp(model.get_prediction(data_tt, tp_tt)).flatten()
        segment.append(segment_x)

    traj_x = np.concatenate(segment, 0)

    plt.scatter(tp, data)
    plt.plot(tp, traj_x)

    for cp in true_cp:
        plt.axvline(x=tp[cp], c='royalblue', lw='4')

    for cp in pred_cp[1:-1]:
        plt.axvline(x=tp[cp], c='orangered', ls='--', lw='2')

    plt.legend([plt.Line2D([0], [0], c='royalblue', lw=4),
                plt.Line2D([0], [0], c='orangered', ls='--', lw=2)],
               ['True CP', 'Predicted CP'])

    plt.show()
