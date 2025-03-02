import sys, os
sys.path.append('..')
import torch
import numpy as np
import nibabel as nib
from mains.trainer import dwi_Trainer
from utils.test_function import read_h5_new, get_config, synthesize_slice

#Define some data specific parameters
min_x, max_x = 8, 136  # the center crop range for the input
min_y, max_y = 8, 166  # the center crop range for the input
pad_x, pad_y = 160, 160  # the pad size of the input image

# #Load the subject data from h5 file
data = '../dataset/test.h5'
affine = np.loadtxt('../dataset/affine_matrix.txt')
print("Affine Matrix:")
print(affine)
np.savetxt('affine_matrix.txt', affine, fmt='%.6f')
header = dwi_img.header
rows, cols, n_slices, n_directions = dwi_data.shape
data_dict = read_h5_new(data, return_dwi=False)

# load the model input candidates
bvec = data_dict['bvec']
bval = data_dict['bval']
b0 = data_dict['b0']
t2 = data_dict['t2']
t1 = data_dict['t1']

#Configure the generator and load a model
config = get_config('../configs/smri2dwi.yaml')
ckpt = os.path.join('../logs/smri2dwi_0102/gen_latest.pt')
trainer = dwi_Trainer(config)
net = trainer.gen_a
state_dict = torch.load(ckpt, map_location=trainer.device)
net.load_state_dict(state_dict['a'])

# Synthesize an image from the gradient table of the data
synth_dwi = np.zeros(dwi_data.shape)

for dir in range(n_directions):
    for s in range(n_slices):
        bvector, bvalue = bvec[dir], bval[dir]
        print("Processing direction:", dir, "slice:", s, "bvector:", bvector, "bvalue:", bvalue)

        synth = synthesize_slice(b0, t2, t1, s, net, bvector, bvalue, pad_x, pad_y, min_x, max_x, min_y, max_y)
        synth_transposed = np.transpose(synth)
        synth_corrected = np.fliplr(synth_transposed)

        synth_dwi[:, :, s, dir] = synth_corrected

synth_img = nib.Nifti1Image(synth_dwi, affine, header)
nib.save(synth_img, '../dataset/test.nii.gz')
