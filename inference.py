from dataset import ShapeNet_Points, ShapeNet_PC
from models import pointnet, pointnet2
from psbody.mesh import Mesh
from loss import chamfer_distance
import os
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device('cuda:0')

shapenet_dir = '/BS/kzhou/static00/shapenet/ShapeNetCore.v1/'
cache_dir = '/BS/kzhou/static00/shapenet/cache'
test_set = ShapeNet_PC('/BS/kzhou2/static00/shapenet/multi/pc_8000.npy', mode=2, num_points=2048)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                          shuffle=False, pin_memory=False,
                                          num_workers=0)


nlat = 128
enc_conv_size = [3, 64, 128, 128, 256]
enc_fc_size = []
dec_fc_size = [256, 256, 2048*3]
enc = pointnet.PointNet(nlat, enc_conv_size)
dec = pointnet.FCDecoder(nlat, dec_fc_size)
#rot_enc = pointnet.RotationEncoder([3, 16, 32, 64], [32, 16, 6]).to(device)
rot_enc = pointnet2.RotationEncoder3()

models = {
    'rot_enc': None,
    'enc': enc,
    'dec': dec,
}

checkpoint_path = '/BS/kzhou2/static00/shapenet_ae/checkpoints3/500.pth'
#checkpoint_path = '/BS/kzhou2/static00/shapenet_ae/multi/498.pth'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
for t in models:
    if models[t]:
        models[t].load_state_dict(checkpoint['m_'+t])
        models[t] = models[t].to(device)

if models['rot_enc']:
    models['rot_enc'].eval()
models['enc'].eval()
models['dec'].eval()

vald_loss = 0

for x in test_loader:
    x = x.to(device)
    with torch.no_grad():
        #R = models['rot_enc'](x)
        z = models['enc'](x)
        y = models['dec'](z)

        with torch.cuda.device(device):
            recon_loss = chamfer_distance(x, y)

        vald_loss += recon_loss.item() * x.size(0)

print(vald_loss / len(test_loader.dataset))