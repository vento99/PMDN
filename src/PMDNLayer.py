import torch
import torch.nn as nn
import torch.nn.functional as F
    
class PMDNLayer(nn.Module):
    def __init__(self, batch_size, num_metadata, num_features): 
        super(PMDNLayer, self).__init__()
        self.batch_size = batch_size
        self.num_metadata = num_metadata # number of metadata (confounders)
        self.cfs = nn.Parameter(torch.randn(batch_size, self.num_metadata), requires_grad=False) # will hold metadata (confounders) for each batch 
        self.beta_mdn = nn.Parameter(torch.zeros(self.num_metadata, num_features), requires_grad=True) # beta for mdn layer (note: no dependence on batch size)
        self.use_labels = False
            
    def forward(self, x):
        # get f
        f = x
        f = f.reshape(x.shape[0], -1)

        # metadata for this batch only
        m_batch = self.cfs 
        
        # calculate m * b
        b = self.beta_mdn
        if self.use_labels:
            f_r = torch.mm(m_batch[:, :], b[:])
        else:
            f_r = torch.mm(m_batch[:, 1:], b[1:])
        
        # determine residual
        residual = f - f_r
        residual = residual.reshape(x.shape)

        # get loss term
        loss_term = (torch.linalg.norm(f - f_r) ** 2)

        return residual, loss_term
    
    