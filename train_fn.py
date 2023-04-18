import numpy as np
import torch
import torch.nn.functional as F
import os, time
import matplotlib.pyplot as plt
from torch import nn, optim
from models import pointnet
from utils import random_rotate_batch, random_rotate_y_batch
from visualize import output_meshes

def unfold_rotenc(data, rotenc, iters=5):
    R_cum = torch.eye(3).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device)
    for _ in range(iters):
        R = rotenc(data.transpose(1, 2).contiguous())
        data = torch.matmul(data, R).detach()
        R_cum = torch.matmul(R_cum, R)
    return R_cum

def train_autoencoder(models, losses, optimizers, data, epoch, opt):
    optimizers['opt'].zero_grad()

    if opt.azimuthal:
        random_rotate = random_rotate_y_batch
    else:
        random_rotate = random_rotate_batch

    if opt.art:
        data_rot_1, rotmat_1 = random_rotate(data)
        data_rot_2, rotmat_2 = random_rotate(data)
        data_rot_3, rotmat_3 = random_rotate(data)
        R_1 = unfold_rotenc(data_rot_1, models['rot_enc'], opt.iters)
        R_2 = unfold_rotenc(data_rot_2, models['rot_enc'], opt.iters)
        R_3 = unfold_rotenc(data_rot_3, models['rot_enc'], opt.iters)

        R = unfold_rotenc(data, models['rot_enc'], opt.iters)

        rotprod_1 = torch.matmul(R, R_1.transpose(1, 2))
        rotprod_2 = torch.matmul(R, R_2.transpose(1, 2))
        rotprod_3 = torch.matmul(R, R_3.transpose(1, 2))

        rot_loss_mse = (F.mse_loss(rotmat_1, rotprod_1) + \
                        F.mse_loss(rotmat_2, rotprod_2) + \
                        F.mse_loss(rotmat_3, rotprod_3)) / 3
        

        z = models['enc'](data, R)
        y = models['dec'](z, R)

        with torch.cuda.device(data.device):
            rot_loss_chamfer = (losses['chamfer'](torch.matmul(data, rotprod_1), data_rot_1) + \
                                losses['chamfer'](torch.matmul(data, rotprod_2), data_rot_2) + \
                                losses['chamfer'](torch.matmul(data, rotprod_3), data_rot_3)) / 3
    elif opt.itn:
        R = unfold_rotenc(data, models['rot_enc'], opt.iters)
        z = models['enc'](data, R)
        y = models['dec'](z, R)
    elif opt.tnet:
        R = models['rot_enc'](data.transpose(1, 2).contiguous())
        z = models['enc'](data, R)
        y = models['dec'](z, torch.inverse(R))
    else:
        data, _ = random_rotate(data)

        z = models['enc'](data)
        y = models['dec'](z)        

    with torch.cuda.device(data.device):
        chamfer_dist = losses['chamfer'](data, y)

    if opt.art:
        loss = chamfer_dist + rot_loss_mse * 0.02 + rot_loss_chamfer * opt.lambda2

    else:
        loss = chamfer_dist
        rot_loss_mse = torch.tensor(0)
        rot_loss_chamfer = torch.tensor(0)

    loss.backward()

    optimizers['opt'].step()

    return chamfer_dist, rot_loss_mse, rot_loss_chamfer

def train_model(models, losses, optimizers, train_loader, vald_loader, device, opt, save_path=None):
    num_epochs = 500
    start_epoch = 1
    vis_step = 500
    log_step = 1
    best_loss = 1000

    ckpt_files = sorted(os.listdir(save_path))
    if opt.resume and len(ckpt_files) > 0:
        ckpt_file = ckpt_files[-1]
        ckpt = torch.load(os.path.join(save_path, ckpt_file), map_location=device)
        for k in models:
            if models[k]:
                models[k].load_state_dict(ckpt['m_'+k])
        for k in optimizers:
            optimizers[k].load_state_dict(ckpt['o_'+k])
        start_epoch = int(ckpt_file.split('.')[0]) + 1

    print('Training started')
    print('azimuthal?', opt.azimuthal)

    for epoch in range(start_epoch, 1+num_epochs):
        t1 = time.time()

        models['enc'].train()
        models['dec'].train()
        if opt.art:
            models['rot_enc'].train()

        train_loader.dataset.resample()

        train_loss_dict = {'chamfer_dist': 0, 'rot_loss_mse': 0, 'rot_loss_chamfer': 0}
        vald_loss_dict = {'chamfer_dist': 0}

        for i, data in enumerate(train_loader):
            data = data.to(device)

            recon_loss, rot_loss_mse, rot_loss_chamfer = train_autoencoder(models, losses, optimizers, data, epoch, opt)

            train_loss_dict['chamfer_dist'] += recon_loss.item() * data.size(0)
            train_loss_dict['rot_loss_mse'] += rot_loss_mse.item() * data.size(0)
            train_loss_dict['rot_loss_chamfer'] += rot_loss_chamfer.item() * data.size(0)

        t2 = time.time()
        print(t2-t1)

        if epoch > 0:
            models['enc'].eval()
            models['dec'].eval()
            if opt.art:
                models['rot_enc'].eval()

            with torch.no_grad():
                for batch_idx, x in enumerate(vald_loader):
                    x = x.to(device)
                    if opt.art:
                        R = unfold_rotenc(x, models['rot_enc'], opt.iters)
                        z = models['enc'](x, R)
                        y = models['dec'](z, R)                    
                    else:
                        z = models['enc'](x)
                        y = models['dec'](z)

                    with torch.cuda.device(device):
                        recon_loss = losses['chamfer'](x, y)

                    vald_loss_dict['chamfer_dist'] += recon_loss.item() * x.size(0)

                    if epoch % vis_step == 0 and batch_idx == 0:
                        x = x.cpu().numpy().reshape(x.shape[0], 1, -1, 3)
                        y = y.cpu().numpy().reshape(y.shape[0], 1, -1, 3)
                        meshes = np.concatenate([x, y], axis=1)
                        output_meshes(meshes, epoch)
                

            if epoch % log_step == 0:
                print('====> Epoch {}/{}: Training'.format(epoch, num_epochs), flush=True)
                
                for term in train_loss_dict:
                    print('\t{} {:.5f}'.format(term, train_loss_dict[term] / len(train_loader.dataset)), flush=True)
                
                vald_loss = vald_loss_dict['chamfer_dist'] / len(vald_loader.dataset)
                print('====> Epoch {}/{}: Validation'.format(epoch, num_epochs), flush=True)
                for term in vald_loss_dict:
                    print('\t{} {:.5f}'.format(term, vald_loss), flush=True)

            if vald_loss < best_loss:
                best_loss = vald_loss
                checkpoint = dict([('m_'+t, models[t].state_dict() if models[t] else None) for t in models])
                checkpoint.update(dict([('o_'+t, optimizers[t].state_dict()) for t in optimizers]))
                checkpoint.update({'torch_rnd': torch.get_rng_state(), 'numpy_rnd': np.random.get_state()})
                torch.save(checkpoint, os.path.join(save_path, '{}.pth'.format(epoch)))

