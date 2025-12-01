import RRAEDy.config
import numpy as np
from RRAEs.AE_classes import RRAE_MLP, RRAE_CNN
from RRAEDy.training_classes import DMD_RRAE_Trainor_class
from RRAEDy.utilities import get_data
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm

folder = "burgers"

trainor_RR = DMD_RRAE_Trainor_class()

# Replace the name below with the path to your saved model
trainor_RR.load_model(f"{folder}_init/RRAE_DMD_{folder}_test.pkl", orig_model_cls=RRAE_MLP)

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
) = get_data(folder)

pr_RR = trainor_RR.model(pre_func_inp(x_train[..., 0:1, :]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[-2]-1)
out = pre_func_out(x_train)
print("RR error train:", np.linalg.norm(pr_RR-out)/np.linalg.norm(out)*100)

pr_RR_test = trainor_RR.model(pre_func_inp(x_test[..., 0:1, :]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[-2]-1)
out = pre_func_out(x_test)
print("RR error test:", np.linalg.norm(pr_RR_test-out)/np.linalg.norm(out)*100)

idxs = [5, 10]
ts = np.arange(0, x_test.shape[1], 5)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np

plt.figure(figsize=(12, 5))

cmap = plt.get_cmap('viridis')
num_ts = len(ts)

for j in range(2):
    plt.subplot(1, 2, j+1)
    idx = idxs[j]
    
    for i, t in enumerate(ts):
        color = cmap(i / max(num_ts - 1, 1))
        
        plt.plot(out[:, t, idx], c=color, linestyle='-')
        plt.plot(pr_RR_test[:, t, idx].T, c=color, linestyle='--')
    
    plt.xlabel(r'$x$', fontsize=16)
    plt.ylabel(r'$u$', fontsize=16)
    plt.title(f'Evolution for sample {j+1}', fontsize=16)

    true_line = mlines.Line2D([], [], color='black', linestyle='-', label='True')
    pred_line = mlines.Line2D([], [], color='black', linestyle='--', label='Prediction')
    color_example = mlines.Line2D([], [], color=cmap(0.5), linestyle='-', label='Time (color-coded)')
    plt.legend(handles=[true_line, pred_line, color_example], loc='upper left')

plt.subplots_adjust(right=0.85)

sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(ts), vmax=max(ts)))
sm.set_array([])
cbar_ax = plt.gcf().add_axes([0.88, 0.15, 0.02, 0.7])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label("Time step", fontsize=14)
plt.show()
