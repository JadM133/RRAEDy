import Dyn_RRAEs.config
import numpy as np
from RRAEs.AE_classes import RRAE_MLP, RRAE_CNN
from Dyn_RRAEs.training_classes import DMD_RRAE_Trainor_class, DMD_Trainor_class
from Dyn_RRAEs.utilities import get_data
import matplotlib.pyplot as plt
import seaborn as sns

# 10 and 29 are indices of max and min Re in the test set

cmap = sns.cubehelix_palette(n_colors=6, start=2, rot=-5, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=True)

def plot_for_paper_fluid_ablation(sample_idx=4, t=None, cmap=cmap):
    
    # choose three timesteps (start, middle, end) along the time axis (assumed at axis -2)
    if t is None:
        n_steps = 3  # default number of timesteps
    elif isinstance(t, int):
        n_steps = t
    else:
        # assume t is an iterable of indices
        t_idxs = list(t)
        n_steps = None

    if n_steps is not None:
        T = x_test.shape[-2]  # length of time axis
        if n_steps <= 1:
            t_idxs = [0]
        else:
            # evenly spaced indices from 0 to T-1 inclusive
            t_idxs = [int(round(i * (T - 1) / (n_steps - 1))) for i in range(n_steps)]
    plt.figure(figsize=(12, 6))
    # turn off axes for all subplots
    # hide ticks and spines but keep axis labels when code calls ax.axis("off")
    import matplotlib.axes as maxes
    _orig_axis = maxes.Axes.axis
    def _axis_keep_labels(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].lower() == "off":
            # remove ticks and ticklabels
            self.set_xticks([])
            self.set_yticks([])
            self.xaxis.set_ticklabels([])
            self.yaxis.set_ticklabels([])
            # hide spines
            for spine in self.spines.values():
                spine.set_visible(False)
            return
        return _orig_axis(self, *args, **kwargs)
    maxes.Axes.axis = _axis_keep_labels

    true_frames = x_test

    # common color scale
    vmin = pr_RR_test.min()
    vmax = pr_RR_test.max()

    # plot: first row = true, second row = predictions
    for col, t in enumerate(t_idxs):
        ax = plt.subplot(3, len(t_idxs), col + 1)
        ax.imshow(true_frames[0, :, :, t, sample_idx].squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        if col == 0:
            ax.set_ylabel("True")
        ax.axis("off")

        ax = plt.subplot(3, len(t_idxs), len(t_idxs) + col + 1)
        ax.imshow(pr_RR_test[0, :, :, t, sample_idx].squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        if col == 0:
            ax.set_ylabel("Dyn-aRRAE")
        ax.axis("off")

        ax = plt.subplot(3, len(t_idxs), 2*len(t_idxs) + col + 1)
        ax.imshow(pr_noRR_test[0, :, :, t, sample_idx].squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        if col == 0:
            ax.set_ylabel("Dyn-AE")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_for_paper_fluid_one(sample_idx=4, t=None, cmap=cmap):
    
    # choose three timesteps (start, middle, end) along the time axis (assumed at axis -2)
    if t is None:
        n_steps = 3  # default number of timesteps
    elif isinstance(t, int):
        n_steps = t
    else:
        # assume t is an iterable of indices
        t_idxs = list(t)
        n_steps = None

    if n_steps is not None:
        T = x_test.shape[-2]  # length of time axis
        if n_steps <= 1:
            t_idxs = [0]
        else:
            # evenly spaced indices from 0 to T-1 inclusive
            t_idxs = [int(round(i * (T - 1) / (n_steps - 1))) for i in range(n_steps)]
    plt.figure(figsize=(12, 6))

    plt.subplot(2, len(t_idxs), 1)

    channel_idx = 0  # visualize first channel by default

    def extract_frame(arr, t, sample=sample_idx, chan=channel_idx):
        # build index tuple: [batch, spatial..., time, channel]
        idxs = [chan]
        # spatial dims are all dims between batch (0) and time (-2)
        for _ in range(1, arr.ndim - 2):
            idxs.append(slice(None))
        idxs.append(t)         # time axis
        idxs.append(sample)      # channel axis
        frame = arr[tuple(idxs)]
        # if there are extra spatial dims collapse by mean until 2D
        while frame.ndim > 2:
            frame = frame.mean(axis=0)
        return frame

    true_frames = [extract_frame(x_test, t) for t in t_idxs]
    pred_frames = [extract_frame(pr_RR_test, t) for t in t_idxs]

    # common color scale
    all_frames = true_frames + pred_frames
    vmin = min(f.min() for f in all_frames)
    vmax = max(f.max() for f in all_frames)

    # plot: first row = true, second row = predictions
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

def plot_for_paper_fluid_one_extrap(sample_idx=4, t=None, cmap=cmap, mul=3):
    
    # choose three timesteps (start, middle, end) along the time axis (assumed at axis -2)
    if t is None:
        n_steps = 3  # default number of timesteps
    elif isinstance(t, int):
        n_steps = t
    else:
        # assume t is an iterable of indices
        t_idxs = list(t)
        n_steps = None

    if n_steps is not None:
        T = x_test.shape[-2]  # length of time axis
        if n_steps <= 1:
            t_idxs = [0]
        else:
            # evenly spaced indices from 0 to T-1 inclusive
            t_idxs = [int(round(i * (T - 1) / (n_steps - 1))) for i in range(n_steps)]
    plt.figure(figsize=(12, 6))
    import pdb; pdb.set_trace()
    import matplotlib.axes as maxes
    _orig_axis = maxes.Axes.axis
    def _axis_keep_labels(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].lower() == "off":
            # remove ticks and ticklabels
            self.set_xticks([])
            self.set_yticks([])
            self.xaxis.set_ticklabels([])
            self.yaxis.set_ticklabels([])
            # hide spines
            for spine in self.spines.values():
                spine.set_visible(False)
            return
        return _orig_axis(self, *args, **kwargs)
    maxes.Axes.axis = _axis_keep_labels

    typic_preds = [pr_RR_test[..., i*int(pr_RR_test.shape[-2]/mul):(i+1)*int(pr_RR_test.shape[-2]/mul), :] for i in range(mul)]
    all_preds = [pr_RR_test_fixed[..., i*int(pr_noRR_test.shape[-2]/mul):(i+1)*int(pr_RR_test.shape[-2]/mul), :] for i in range(mul)]

    # common color scale
    vmin = pr_RR_test.min()
    vmax = pr_RR_test.max()

    # plot: first row = true, second row = predictions
    for col, t in enumerate(t_idxs):
        for i in range(mul):
            ax = plt.subplot(mul, len(t_idxs), i * len(t_idxs) + col + 1)
            ax.imshow(all_preds[i][0, :, :, t, sample_idx], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if col == 0:
                match i:
                    case 0:
                        ax.set_ylabel(r"First $540^\circ$")
                    case 1:
                        ax.set_ylabel(r"Second $540^\circ$")
                    case 2:
                        ax.set_ylabel(r"Third $540^\circ$")
            ax.axis("off")

    plt.tight_layout()
    plt.show()

folder = "circular_gaussian"

orig_model_cls = RRAE_CNN

trainor_RR = DMD_RRAE_Trainor_class()
trainor_RR.load_model(f"{folder}_res/{folder}_RR/RRAE_DMD_{folder}_test.pkl", orig_model_cls=orig_model_cls)
reg = max(1, np.max(np.abs(np.linalg.eig(trainor_RR.W)[0])))
# print(reg)
# trainor_RR.W = trainor_RR.W / reg

trainor_noRR = DMD_RRAE_Trainor_class()
trainor_noRR.load_model(f"{folder}_res/{folder}_noRR/RRAE_DMD_{folder}_test.pkl", orig_model_cls=orig_model_cls)
reg = max(1, np.max(np.abs(np.linalg.eig(trainor_noRR.W)[0])))
# print(reg)
# trainor_noRR.W = trainor_noRR.W / reg

trainor_RR_fixed = DMD_RRAE_Trainor_class()
trainor_RR_fixed.load_model(f"{folder}_res/{folder}_RR_fixed/RRAE_DMD_{folder}_test.pkl", orig_model_cls=orig_model_cls)
reg = max(1, np.max(np.abs(np.linalg.eig(trainor_RR_fixed.W)[0])))
# print(reg)
# trainor_RR_fixed.W = trainor_RR_fixed.W / reg

trainor_RR_nodmd = DMD_RRAE_Trainor_class()
trainor_RR_nodmd.load_model(f"{folder}_res/{folder}_RR_nodmd/RRAE_DMD_{folder}_test.pkl", orig_model_cls=orig_model_cls)

(
    x_train,
    x_test,
    p_train,
    p_test,
    y_train,
    y_test,
    pre_func_inp,
    pre_func_out,
    kwargs,
) = get_data(folder, train_size=200, test_size=1000, T=20, folder="../")
print(x_train.shape)

mul = 3
pr_RR_test = trainor_RR.model(pre_func_inp(x_test[..., 0:1, -10:]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[-2]*mul-1)
out = pre_func_out(x_test[..., -10:])
# print("RR error test:", np.linalg.norm(pr_RR_test-out)/np.linalg.norm(out)*100)

pr_noRR_test = trainor_noRR.model(pre_func_inp(x_test[..., 0:1, -10:]), p=None, apply_W=trainor_noRR.W, steps=x_test.shape[-2]*mul-1, RR=False)
out = pre_func_out(x_test[..., -10:])
# print("noRR error test:", np.linalg.norm(pr_noRR_test-out)/np.linalg.norm(out)*100)

pr_RR_test_fixed = trainor_RR_fixed.model(pre_func_inp(x_test[..., 0:1, -10:]), p=None, apply_W=trainor_RR_fixed.W, apply_basis=trainor_RR_fixed.basis, steps=x_test.shape[-2]*mul-1)
out = pre_func_out(x_test[..., -10:])
# print("RR fixed error test:", np.linalg.norm(pr_RR_test_fixed-out)/np.linalg.norm(out)*100)

pr_RR_test_nodmd = trainor_RR_nodmd.model(pre_func_inp(x_test[..., 0:1, -10:]), p=None, apply_W=trainor_RR_nodmd.W, apply_basis=trainor_RR_nodmd.basis, steps=x_test.shape[-2]-1)
out = pre_func_out(x_test[..., -10:])
print("RR nodmd error test:", np.linalg.norm(pr_RR_test_nodmd-out)/np.linalg.norm(out)*100)

plot_for_paper_fluid_one_extrap(sample_idx=6, t=8, cmap=cmap, mul=3)

plot_for_paper_fluid_one(sample_idx=4, t=3, cmap=cmap)

plot_for_paper_fluid_ablation(sample_idx=6, t=8, cmap=cmap)

import pdb; pdb.set_trace()