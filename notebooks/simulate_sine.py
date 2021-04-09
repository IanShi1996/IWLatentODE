import numpy as np


def transform_sine(tp, freq, amp, phase=0, n_samp=1, noise=0.):
    """Evaluate sine wave at given timepoints.

    Args:
        tp (np.ndarray): Timepoints at which to evaluate sine wave.
        freq (float): Frequency of sine wave.
        amp (float): Amplitude of sine wave.
        phase (float, optional): Phase of sine wave in radians.
        n_samp (int, optional): Number of sine waves to generate.
        noise (float, optional): Std of normal noise to add to sine wave.

    Returns:
        np.ndarray: Sine wave evaluated at given timepoints.
    """
    data = amp * np.sin(tp * freq + phase * np.pi)

    data = np.repeat(np.expand_dims(data, 0), n_samp, 0)
    noise = np.random.normal(0, noise, (n_samp, data.shape[1]))

    return data + noise


def generate_sine_linear(n_samp, freq, amp, phase, end, noise_std=0.):
    """Generate sine wave with linear time steps.

    Args:
        n_samp (int): Number of samples in trajectory.
        freq (float): Frequency of sine wave.
        amp (float): Amplitude of sine wave.
        phase (float): Phase of sine wave in radians.
        end (float): Last timepoint of trajectory.
        noise_std (float, optional): Standard deviation of noise added to sine.
    Returns:
        np.ndarray: Sine wave observation times and data.
    """
    tp = np.linspace(0, end, n_samp)
    data = transform_sine(tp, freq, amp, phase, 1, noise_std)

    return tp, data


def generate_sine_uniform(n_samp, freq, amp, phase, end, noise_std=0.):
    """Generate sine wave with uniformly sampled time steps.
    Args:
        n_samp (int): Number of samples in trajectory.
        freq (float): Frequency of sine wave.
        amp (float): Amplitude of sine wave.
        phase (float): Phase of sine wave in radians.
        end (float): Last timepoint of trajectory.
        noise_std (float, optional): Standard deviation of noise added to sine.
    Returns:
        np.ndarray: Sine wave observation times and data.
    """
    tp = np.sort(np.random.uniform(0, end, n_samp))
    data = transform_sine(tp, freq, amp, phase, 1, noise_std)

    return tp, data


def generate_disjoint_timepoints(n_samp, start, end):
    """Generate three sets of disjoint timepoints.

    Generates three sets of disjoint timepoints. Timepoints are drawn from an
    uniform distribution. Intended for usage as training/validation/test set
    observation times for model training.

    Args:
        n_samp (int, int, int): Tuple specifying number of samples. Entries
            specify number of train/val/test samples respectively.
        start (float): First timepoint.
        end (float): Last timepoint.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Timepoints in train/val/test set.
    """
    n_total = sum(n_samp)
    timepoints = np.random.uniform(start, end, n_total)

    train_time = np.sort(timepoints[:n_samp[0]])
    val_time = np.sort(timepoints[n_samp[0]:n_samp[0]+n_samp[1]])
    test_time = np.sort(timepoints[n_samp[0]+n_samp[1]:])

    return train_time, val_time, test_time


class SineSetGenerator:
    """Generates three sets of sine waves according to parameterization.

    Intended for use as train/val/test datasets for Latent Node models.
    Sine waves are generated using amplitude and frequency sampled from a
    uniform distribution bounded by input parameters. Each trajectory will
    share timepoints within a specific set. Timepoints can be made evenly
    spaced, or sampled under a uniform distribution. Differences in phase
    (and thus initial condition) can be included.

    Additional features such as changing data generation scheme to favour
    specific trajectories, or adding more sine wave parameters such as phase
    could be added in the future.

    Attributes:
        n_traj (int, int, int): Number of trajectories in train/val/test set.
        n_samp (int, int, int): Number of samples per trajectory in
            train/val/test set.
        amp (float, float): Mininum and maximum amplitudes.
        freq (float, float): Minimum and maximum frequencies.
        phase (boolean): Whether phase variation should be included.
        start (float): Initial timepoint for trajectories.
        end (float): Final timepoint for trajectories.
        noise (float): Standard deviation of simulated noise.
        tp_generation (string, string, string): Tuple representing timepoint
            generation strategy. Can be either L for linear or U for uniform.
    """

    def __init__(self, params):
        """Initialize SineSetGenerator.

        Expects a dictionary of many parameters. Currently does not check
        validity of these arguments, nor is backwards compatible.

        Args:
            params (dict): Dictionary of sine wave parameters.
        """
        self.n_traj = params['n_traj']
        self.n_samp = params['n_samp']

        self.amp = params['amp']
        self.freq = params['freq']
        self.phase = params['phase']
        self.start = params['start']
        self.end = params['end']
        self.noise = params['noise']

        self.tp_generation = params['tp_generation']

        self.generate_timepoints()
        self.generate_data()

    def generate_timepoints(self):
        """Generate observation times.

        Timepoints are shared across a set of data. Timepoints are
        sampled according to a uniform distribution by default, but can
        be switched to a linspace via the tp_generation flag.

        This default behavior should be modified when other timepoint
        generation schemes are required.
        """
        tp = list(generate_disjoint_timepoints(self.n_samp, self.start,
                                               self.end))
        for i in range(len(self.tp_generation)):
            if self.tp_generation[i] == 'L':
                tp[i] = np.linspace(self.start, self.end, self.n_traj[i])

        tp = tuple(tp)
        self.train_time, self.val_time, self.test_time = tp

    def generate_data(self):
        """Generate data by uniformly sampling frequency and amplitude."""
        train_data = []
        val_data = []
        test_data = []

        total_traj = sum(self.n_traj)

        freqs = np.random.uniform(self.freq[0], self.freq[1], total_traj)
        amps = np.random.uniform(self.amp[0], self.amp[1], total_traj)

        if self.phase:
            phase = np.random.uniform(0, 2, total_traj)
        else:
            phase = [0] * total_traj

        for i in range(self.n_traj[0]):
            train_d = transform_sine(self.train_time, freqs[i], amps[i],
                                     phase[i], noise=self.noise)
            train_data.append(train_d)

        for i in range(self.n_traj[0], self.n_traj[0]+self.n_traj[1]):
            val_d = transform_sine(self.val_time, freqs[i], amps[i],
                                   phase[i], noise=self.noise)
            val_data.append(val_d)

        for i in range(self.n_traj[0]+self.n_traj[1], total_traj):
            test_d = transform_sine(self.test_time, freqs[i], amps[i],
                                    phase[i], noise=self.noise)
            test_data.append(test_d)

        train_data = np.stack(train_data, 0)
        train_data = np.concatenate(train_data, 0)

        val_data = np.stack(val_data, 0)
        val_data = np.concatenate(val_data, 0)

        test_data = np.stack(test_data, 0)
        test_data = np.concatenate(test_data, 0)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def get_train_set(self):
        """Retrieve training timepoints and data.

        Returns:
            (np.ndarray, np.ndarray): Training timepoints and data.
        """
        return self.train_time, self.train_data

    def get_test_set(self):
        """Retrieve test timepoints and data.

        Returns:
            (np.ndarray, np.ndarray): Test timepoints and data.
        """
        return self.test_time, self.test_data

    def get_val_set(self):
        """Retrieve validation timepoints and data.

        Returns:
            (np.ndarray, np.ndarray): Validation timepoints and data.
        """
        return self.val_time, self.val_data

