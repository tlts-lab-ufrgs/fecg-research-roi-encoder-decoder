import numpy as np

def add_baseline_wandering(x, num_components=5, amplitude=0.1, fs=1000):
    t = np.arange(len(x)) / fs
    baseline_wandering = np.zeros_like(x)

    for _ in range(num_components):
        frequency = np.random.uniform(low=0.05, high=0.5)  # Random low frequency
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        component = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        baseline_wandering += component

    x_with_baseline = x + baseline_wandering
    return x_with_baseline