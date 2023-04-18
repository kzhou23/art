import torch
import numpy as np
import cv2


def mesh_rotate(x, axis, rad):
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    mat = cv2.Rodrigues(axis.reshape(3, 1)*rad)

    return x.dot(mat[0].T)


def random_rotate_np(x):
    aa = np.random.randn(3)
    theta = np.sqrt(np.sum(aa**2))
    k = aa / np.maximum(theta, 1e-6)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.matmul(K, K)
    return x @ R, R

def random_rotate_x_np(x, low=0, high=np.pi*2):
    theta = np.random.uniform(low, high)
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return x @ R

def random_rotate_y_np(x):
    theta = np.random.uniform(0, np.pi*2)
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    return x @ R

def rotate_z_np(x, theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return x @ R
    
def rotate_x_np(x, theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return x @ R

def rotate_y_np(x, theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return x @ R

def random_rotate_batch(x):
    aa = torch.randn((3,), dtype=torch.float32)
    theta = torch.sqrt(torch.sum(aa**2))
    k = aa / (theta + 1e-6)
    K = torch.tensor([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], device=x.device)
    R = torch.eye(3, device=x.device) + torch.sin(theta)*K + (1-torch.cos(theta))*torch.mm(K, K)
    R = R.unsqueeze(0).repeat(x.size(0), 1, 1)
    return torch.matmul(x, R), R


def surf_normalize(x, f):
    cross = np.cross((x[..., f[:, 1], :]-x[..., f[:, 0], :]), (x[..., f[:, 2], :]-x[..., f[:, 0], :]))
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = .5 * np.sum(norm, axis=-2, keepdims=True)
    return x / areas

def pca_align(x, temp_axes):
    '''
    x: B*N*3
    template: 3*3
    '''
    x = x - x.mean(dim=1, keepdim=True)
    x_cov = torch.matmul(x.transpose(1, 2), x) / (x.size(1)-1)
    _, v = torch.symeig(x_cov, eigenvectors=True)
    vec_a = v[:, :, 2]
    vec_b = v[:, :, 1]
    cross_prod = torch.cross(vec_a, vec_b)
    x_axes = torch.stack([vec_a, vec_b, cross_prod], dim=1)
    R = torch.matmul(x_axes, temp_axes)
    return torch.matmul(x, R), R