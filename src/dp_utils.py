import torch
import numpy as np

def add_differential_privacy_to_updates(updates, clipping, epsilon, delta, device):
    #clip updates
    clipped_updates = clip_updates(updates, clipping)
    # compute sigma
    noise_std = sigma(epsilon, delta, clipping)
    # add noise to updates
    dp_updates = add_noise(clipped_updates, noise_std, device)
    return dp_updates
    
def clip_updates(updates, clipping):
    total_norm = 0.0

    for update in updates:
        total_norm += torch.sum(update ** 2)
    total_norm = torch.sqrt(total_norm)

    # clipping
    factor = min(1,clipping / total_norm)
    return [update * factor for update in updates]

def sigma(epsilon, delta, clipping):
    c = np.sqrt(2 * np.log(1.25 / delta))
    return c * clipping / epsilon

def add_noise(updates, sigma, device):
    return [update + torch.normal(mean=0, std=sigma, size=update.shape).to(device) for update in updates]