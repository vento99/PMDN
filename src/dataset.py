import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
import scipy.stats as st

"""
File uses the same dataset class and data generation function as the repository: https://github.com/mlu355/MetadataNorm  
"""

# # Simulating Synthetic Images
# The training images of two groups are simulated. Each image contains 4 Gaussian distribution density functions. Let the 4 standard deviations be
#
# |  $\sigma_1$ | $\sigma_2$  |
#
# |  $\sigma_3$ | $\sigma_4$  |
#
# The 4 Gaussians are constructed such that
#
# 1. two diagonal Gaussians $\sigma_1,\sigma_4$ are linked to a factor of interest $mf$ (e.g. true effect between two classes)
# 2. two off-diagonal Gaussians $\sigma_2,\sigma_3$ are linked to two different confounding factors $cf_1, cf_2$.

## Simulate Data
def generate_data(N, seed=4201):    
    np.random.seed(seed)
    
    labels = np.zeros((N*2,))
    labels[N:] = 1

    # 2 confounding effects between 2 groups
    cf1 = np.zeros((N*2,))
    cf2 = np.zeros((N*2,))
    cf1[:N] = np.random.uniform(1,4,size=N) 
    cf1[N:] = np.random.uniform(3,6,size=N) 
    cf2[:N] = np.random.uniform(1,4,size=N) 
    cf2[N:] = np.random.uniform(3,6,size=N)

    # 2 major effects between 2 groups
    np.random.seed(seed+1)
    mf = np.zeros((N*2,))
    mf[:N] = np.random.uniform(1,4,size=N) 
    mf[N:] = np.random.uniform(3,6,size=N)
    
    # simulate images
    d = int(32)
    dh = d//2
    x = np.zeros((N*2,d,d,1)) 
    y = np.zeros((N*2,)) 
    y[N:] = 1
    for i in range(N*2):
        x[i,:dh,:dh,0] = gkern(kernlen=d//2, nsig=5)*mf[i]#nsig=mf[i]) 
        x[i,dh:,:dh,0] = gkern(kernlen=d//2, nsig=5)*cf1[i]
        #x[i,:d//2,d//2:,0] = gkern(kernlen=16, nsig=5)*cf2[i] # only use one confounder for now
        x[i,dh:,dh:,0] = gkern(kernlen=d//2, nsig=5)*mf[i]#nsig=mf[i]) 
        x[i] = x[i] + np.random.normal(0,0.01,size=(d,d,1)) # random noise
        
    return labels, cf1, cf2, mf, x, y

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

class SyntheticDataset(Dataset):
    """Synthetic dataset."""

    def __init__(self, imgs, labels, cfs, transform=None):
        """
        Args:
            imgs (array of images): array of input images
            labels (array of [0, 1]): array of labels
            cfs (array of cfs): array of cfs 
            transform (cfs): Optional transform to be applied
                on a sample.
        """
        self.imgs = imgs
        self.labels = labels
        self.cfs = cfs
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]

        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        cf = self.cfs[idx]
        datum = {'image': image, 'label': int(label), 'cfs': cf}
        return datum