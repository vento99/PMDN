import torch
import torch.nn as nn
import numpy as np
import dcor
import copy
from tqdm import tqdm

from dataset import generate_data, SyntheticDataset
from model import PMDN_Model


def train_and_validate(model, train_loader, val_loader, device, epochs, lr_beta, lr_model):
    # Separate beta params from model params
    model_params, beta_params = [], []
    for name, param in model.named_parameters():
        if "cf" in name: # not a trainable parameter
            continue
        elif "beta_mdn" in name:
            beta_params.append(param)
        else:
            model_params.append(param)
    
    # Tune the learning rates as much as you'd like! 
    model_optimizer = torch.optim.Adam(model_params, lr=lr_model) 
    beta_optimizer = torch.optim.Adam(beta_params, lr=lr_beta) 
    lr_scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min', min_lr=1e-6, factor=0.5)
    lr_scheduler_beta = torch.optim.lr_scheduler.ReduceLROnPlateau(beta_optimizer, 'min', min_lr=1e-6, factor=0.5) 
    criterion = torch.nn.BCELoss()

    # progress bar 
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        # Train
        model.train()
        train_bs = train_loader.batch_size
        for i, sample_batched in enumerate(train_loader):
            data = sample_batched['image'].float().to(device)
            target = sample_batched['label'].float().to(device)
            cf1_batch = sample_batched['cfs'].float()

            # Metadata / Confounders for this batch
            X_batch = np.zeros((train_bs,3))
            X_batch[:,0] = target.cpu().detach().numpy()
            X_batch[:,1] = cf1_batch.cpu().detach().numpy()
            X_batch[:,2] = np.ones((train_bs,))
            with torch.no_grad():
                model.cfs = nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False) # set the metadata (confounders) for this batch

            # forward pass 1 (update betas)
            model.use_pmdn_labels = True
            y_pred, flat, fc_norm = model(data)
            beta_loss = model.loss_terms 
            beta_loss.backward()
            beta_optimizer.step()
            model.zero_grad()

            # forward pass 2 (update model params)
            model.use_pmdn_labels = False
            y_pred, flat, fc_norm = model(data)
            model_loss = criterion(y_pred, target.unsqueeze(1))
            model_loss.backward()
            model_optimizer.step()
            model.zero_grad()
        
        # Validate
        model.eval()
        val_bs = val_loader.batch_size
        total_beta_loss, total_model_loss = 0, 0
        fc_norm_val, cf_val, label_val, pred_val = [], [], [], []
        for i, sample_batched in enumerate(val_loader):
            data = sample_batched['image'].float().to(device)
            target = sample_batched['label'].float().to(device)
            cf1_batch = sample_batched['cfs'].float()

            # Metadata / Confounders for this batch
            X_batch = np.zeros((val_bs,3))
            X_batch[:,0] = target.cpu().detach().numpy()
            X_batch[:,1] = cf1_batch.cpu().detach().numpy()
            X_batch[:,2] = np.ones((val_bs,))

            with torch.no_grad():
                # one forward pass through the model
                model.cfs = nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False) 
                model.use_pmdn_labels = False
                y_pred, flattened, fc_norm = model(data)

                model_loss = criterion(y_pred, target.unsqueeze(1))
                beta_loss = model.loss_terms 

                total_beta_loss += beta_loss.item()
                total_model_loss += model_loss.item()

                fc_norm_val.append(fc_norm)
                cf_val.append(cf1_batch)
                label_val.append(target.cpu())
                pred_val.append(torch.round(y_pred).cpu())
                
        lr_scheduler_beta.step(total_beta_loss / len(val_loader))
        lr_scheduler_model.step(total_model_loss / len(val_loader))

        # Calculate correlations
        epoch_targets = np.concatenate(label_val, axis=0)
        epoch_preds = np.concatenate(pred_val, axis=0)
        epoch_layer_norm = np.concatenate(fc_norm_val, axis=0)
        epoch_cf1 = np.concatenate(cf_val, axis=0)

        i0_val = np.where(epoch_targets == 0)[0]
        i1_val = np.where(epoch_targets == 1)[0]

        dc0_norm_val = dcor.u_distance_correlation_sqr(epoch_layer_norm[i0_val], epoch_cf1[i0_val])
        dc1_norm_val = dcor.u_distance_correlation_sqr(epoch_layer_norm[i1_val], epoch_cf1[i1_val])

        # Calcculate accuracy 
        val_acc = np.sum(epoch_targets == epoch_preds.reshape(-1)) / len(epoch_targets)

        # Update progress bar
        pbar.set_postfix(
            val_acc='%.3f' % val_acc,
            val_corr_feature_0='%.3f' % dc0_norm_val,
            val_corr_feature_1='%.3f' % dc1_norm_val
        )

    final_model = copy.deepcopy(model)
    return final_model

def test(final_model, test_loader, device):
    final_model.eval()
    fc_norm_test, cf_test, label_test, pred_test = [], [], [], []
    test_bs = test_loader.batch_size
    for i, sample_batched in enumerate(test_loader):
        data = sample_batched['image'].float().to(device)
        target = sample_batched['label'].float().to(device)
        cf1_batch = sample_batched['cfs'].float()

        X_batch = np.zeros((test_bs,3))
        X_batch[:,0] = target.cpu().detach().numpy()
        X_batch[:,1] = cf1_batch.cpu().detach().numpy()
        X_batch[:,2] = np.ones((test_bs,))

        with torch.no_grad():
            final_model.cfs = nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)
            final_model.use_pmdn_labels = False
            y_pred, flattened, fc_norm = final_model(data)

            fc_norm_test.append(fc_norm)
            cf_test.append(cf1_batch)
            label_test.append(target.cpu())
            pred_test.append(torch.round(y_pred).cpu())

    # Calculate correlations
    epoch_targets = np.concatenate(label_test, axis=0)
    epoch_preds = np.concatenate(pred_test, axis=0)
    epoch_layer_norm = np.concatenate(fc_norm_test, axis=0)
    epoch_cf1 = np.concatenate(cf_test, axis=0)

    i0_val = np.where(epoch_targets == 0)[0]
    i1_val = np.where(epoch_targets == 1)[0]

    dc0_norm_val = dcor.u_distance_correlation_sqr(epoch_layer_norm[i0_val], epoch_cf1[i0_val])
    dc1_norm_val = dcor.u_distance_correlation_sqr(epoch_layer_norm[i1_val], epoch_cf1[i1_val])

    # Calcculate accuracy 
    test_acc = np.sum(epoch_targets == epoch_preds.reshape(-1)) / len(epoch_targets)

    return dc0_norm_val, dc1_norm_val, test_acc



if __name__ == '__main__':
    # device -- cuda recommended 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # Params
    N = 1000 # Number of samples per each of the two group s
    num_metadata = 3 # Num of Metadata (Confounders)
    epochs = 100
    num_runs = 100
    batch_size = 200
    lr_beta = 0.045
    lr_model = 0.00001

    # Generate the data and move channels after batch so we have (N, channels, h, w)
    labels, cf1, _, _, x, y = generate_data(N, seed=1)
    labels_val, cf1_val, _, _, x_val, y_val = generate_data(N, seed=42)
    labels_test, cf1_test, _, _, x_test, y_test = generate_data(N, seed=3)
    x_new = np.swapaxes(x, 1, 3)
    x_new_val = np.swapaxes(x_val, 1, 3)
    x_new_test = np.swapaxes(x_test, 1, 3)

    # Get data and dataloaders
    train_set = SyntheticDataset(x_new, labels, cf1)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_set = SyntheticDataset(x_new_val, labels_val, cf1_val)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_set = SyntheticDataset(x_new_test, labels_test, cf1_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=20, shuffle=True, pin_memory=True)

    f0_corr, f1_corr, acc = 0, 0, 0
    for _ in range(num_runs):
        model = PMDN_Model(batch_size, num_metadata).to(device)
        final_model = train_and_validate(model, train_loader, val_loader, device, epochs, lr_beta, lr_model)
        
        f0_corr_temp, f1_corr_temp, acc_temp = test(final_model, test_loader, device)
        f0_corr += f0_corr_temp
        f1_corr += f1_corr_temp
        acc += acc_temp

    print("Overall Correlations:", f0_corr / num_runs, f1_corr / num_runs)
    print("Overall Accuracy:", acc / num_runs)

