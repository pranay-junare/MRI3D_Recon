#3d visualization

import numpy as np
import nibabel as nib
import plotly.graph_objects as go

#read in image files
TRAIN_DATASET_PATH = 'brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
b1_flair = test_image_flair.get_fdata()
test_image_seg=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')
b1_tumor = test_image_seg.get_fdata()

def get_vals(volume, threshold = .01, nsamples = 50000):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    mask = volume > threshold
    x, y, z = np.where(mask)
    values = volume[mask]
    if len(x) > nsamples:
        indices = np.random.choice(len(x), nsamples, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
        values = values[indices]
    return x, y, z, values

def visualize_tumor(brain, tumor):
    
    xb, yb, zb, valsb = get_vals(brain, nsamples=100000)
    xt, yt, zt, valst = get_vals(tumor, threshold=.1)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xb, y=yb, z=zb,
        mode='markers',
        marker=dict(
            size=1,
            color=valsb,
            colorscale='Gray',
            opacity=0.3
        ), name = "Brain"
    ))
    
    fig.add_trace(go.Scatter3d(
        x=xt, y=yt, z=zt,
        mode='markers',
        marker=dict(
            size=1,
            color=valst,
            colorscale='Magma',
            opacity=0.8
        ), name = "Tumor"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=700,
        margin=dict(r=10, l=10, b=10, t=10)
    )

    fig.show()

    return

visualize_tumor(b1_flair, b1_tumor)