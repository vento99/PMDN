import torch
import torch.nn as nn
import torch.nn.functional as F
from PMDNLayer import PMDNLayer

class PMDN_Model(nn.Module):
    def __init__(self, batch_size, num_metadata):
        super(PMDN_Model, self).__init__()
        
        # PMDN inits
        self.num_metadata = num_metadata
        self.batch_size = batch_size
        self.cfs = nn.Parameter(torch.randn(batch_size, num_metadata), requires_grad=False) # confounders per batch
        self.loss_terms = 0 # aggregates all losses

        # PMDN layers
        self.PMDN1 = PMDNLayer(self.batch_size, num_metadata, 16*28*28)
        self.PMDN2 = PMDNLayer(self.batch_size, num_metadata, 32*24*24)
        self.PMDN_fc = PMDNLayer(self.batch_size, num_metadata, 84)

        # Boolean which says if PMDN layer should use labels
        self.use_pmdn_labels = False

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)

        # FC layers
        self.fc_last = nn.Linear(84, 1)
        self.fc = nn.Linear(18432, 84)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Layernorm layers
        self.layernorm1 = nn.LayerNorm([16, 28, 28])
        self.layernorm2 = nn.LayerNorm([32, 24, 24])
        self.ln_fc = nn.LayerNorm([84])
        
    def forward(self, x):
        self.loss_terms = 0

        # First Conv + PMDN Block
        x = self.conv1(x)
        x = self.layernorm1(x)
        self.PMDN1.cfs = self.cfs
        self.PMDN1.use_labels = self.use_pmdn_labels
        x, loss_term = self.PMDN1(x)
        self.loss_terms += loss_term
        x = F.relu(x)

        # Second Conv + PMDN Block
        x = self.conv2(x)
        x = self.layernorm2(x)
        self.PMDN2.cfs = self.cfs
        self.PMDN2.use_labels = self.use_pmdn_labels
        x, loss_term = self.PMDN2(x)
        self.loss_terms += loss_term 
        x = F.relu(x)

        # FC Block
        x = x.view(-1, 18432)
        flat = x.cpu().detach().numpy().copy()
        x = self.fc(x)   
        x = self.ln_fc(x)
        self.PMDN_fc.cfs = self.cfs
        self.PMDN_fc.use_labels = self.use_pmdn_labels
        x, loss_term = self.PMDN_fc(x)
        self.loss_terms += loss_term
        fc_norm = x.cpu().detach().numpy()
        x = F.relu(x)

        # Final FC + Sigmoid
        x = self.fc_last(x)
        x = self.sigmoid(x)

        # Avereage loss
        self.loss_terms = 1/3 * self.loss_terms

        return x, flat, fc_norm