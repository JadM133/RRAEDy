import RRAEDy.config
import numpy as np
from RRAEs.AE_classes import RRAE_MLP, RRAE_CNN
from RRAEDy.training_classes import DMD_RRAE_Trainor_class
from RRAEDy.utilities import get_data
import matplotlib.pyplot as plt
import seaborn as sns

cmap = sns.cubehelix_palette(n_colors=6, start=1, rot=-3, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=True)

def plot_for_paper_fluid(sample_idx=10, t=None, cmap=cmap):
    
    if t is None:
        t_idxs = [1, x_test.shape[-2] // 3, 2*x_test.shape[-2] // 3, x_test.shape[-2] - 1]
    else:
        t_idxs = t

    plt.subplot(2, len(t_idxs), 1)

    channel_idx = 0 

    def extract_frame(arr, t, sample=sample_idx, chan=channel_idx):
        idxs = [chan]
        for _ in range(1, arr.ndim - 2):
            idxs.append(slice(None))
        idxs.append(t)
        idxs.append(sample)
        frame = arr[tuple(idxs)]
        while frame.ndim > 2:
            frame = frame.mean(axis=0)
        return frame

    true_frames = [extract_frame(x_test, t) for t in t_idxs]
    pred_frames = [extract_frame(pr_RR_test, t) for t in t_idxs]

    all_frames = true_frames + pred_frames
    vmin = min(f.min() for f in all_frames)
    vmax = max(f.max() for f in all_frames)

    for col, t in enumerate(t_idxs):
        ax = plt.subplot(2, len(t_idxs), col + 1)
        ax.imshow(true_frames[col].squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"True t={t}")
        ax.axis("off")

        ax = plt.subplot(2, len(t_idxs), len(t_idxs) + col + 1)
        ax.imshow(pred_frames[col].squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"Pred t={t}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

folder = "fluid"

trainor_RR = DMD_RRAE_Trainor_class()
# Replace the name below with the path to your saved model
trainor_RR.load_model(f"{folder}_init/RRAE_DMD_{folder}_test.pkl", orig_model_cls=RRAE_CNN)

import pickle

# Replace the name below with the path to your data generated using fluid_data_gen.py
with open("fluid_res/all_test_data.pkl", "rb") as f:
    x_test = pickle.load(f)

x_test = x_test[..., ::5, :]
x_train = x_test  # Using test data as train data for this example
p_train = None
p_test = None
pre_func_inp = lambda x:x
pre_func_out = lambda x:x
kwargs = {}

pr_RR = trainor_RR.model(pre_func_inp(x_train[..., 0:1, :]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[-2]-1)
out = pre_func_out(x_train)
print("RR error train:", np.linalg.norm(pr_RR-out)/np.linalg.norm(out)*100)

pr_RR_test = trainor_RR.model(pre_func_inp(x_test[..., 0:1, :]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[-2]-1)
out = pre_func_out(x_test)
print("RR error test:", np.linalg.norm(pr_RR_test-out)/np.linalg.norm(out)*100)



import pdb; pdb.set_trace()