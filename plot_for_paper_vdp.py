import Dyn_RRAEs.config
import numpy as np
from RRAEs.AE_classes import RRAE_MLP, RRAE_CNN
from Dyn_RRAEs.training_classes import DMD_RRAE_Trainor_class, DMD_Trainor_class
from Dyn_RRAEs.utilities import get_data
import matplotlib.pyplot as plt

folder = "van_der_pol"

trainor_RR = DMD_RRAE_Trainor_class()
trainor_RR.load_model(f"{folder}_res/{folder}_RR_3/RRAE_DMD_{folder}_test.pkl", orig_model_cls=RRAE_MLP)

todo = 'loss_plot' # Choose from: 'eig', 'pred_error', 'interp_components', 'loss_plot', 'pred_long', 'just_pass

(
    _x_train,
    _x_test,
    p_train,
    p_test,
    _y_train,
    _y_test,
    pre_func_inp,
    pre_func_out,
    kwargs,
) = get_data(folder, train_size=200, test_size=1, T=20, folder="../")

x_train = _x_train[:, ::5]
y_train = _y_train[:, ::5]
x_test = _x_test[:, ::5]
y_test = _y_test[:, ::5]

pr_RR = trainor_RR.model(pre_func_inp(x_train[..., 0:1, :]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[1]-1)
out = pre_func_out(x_train)
print("RR error train:", np.linalg.norm(pr_RR-out)/np.linalg.norm(out)*100)

pr_RR_test = trainor_RR.model(pre_func_inp(x_test[..., 0:1, :]), p=None, apply_W=trainor_RR.W, apply_basis=trainor_RR.basis, steps=x_test.shape[1]-1)
out = pre_func_out(x_test)
print("RR error test:", np.linalg.norm(pr_RR_test-out)/np.linalg.norm(out)*100)

idxs = [10]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i, idx in enumerate(idxs):
    if i == 0:
        plt.plot(out[0, :, idx], label='true', c="k")
        plt.plot(pr_RR_test[0, :, idx].T, '--', label='pred', c="r")
        plt.plot(out[1, :, idx], label='_', c="k")
        plt.plot(pr_RR_test[1, :, idx].T, '--', label='_', c="r")
    else:
        plt.plot(out[:, :, idx].T, label="_", c="k")
        plt.plot(pr_RR_test[:, :, idx].T, label="_", c="r")
plt.xlabel('Time', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.title('Time series', fontsize=16)
plt.legend()
plt.subplot(1, 2, 2)
for i, idx in enumerate(idxs):
    if i == 0:
        plt.plot(out[0, :, idx], out[1, :, idx], label='true', c="k")
        plt.plot(pr_RR_test[0, :, idx], pr_RR_test[1, :, idx], '--', label='pred', c="r")
    else:
        plt.plot(out[0, :, idx], out[1, :, idx], label="_", c="k")
        plt.plot(pr_RR_test[0, :, idx], pr_RR_test[1, :, idx], label="_", c="r")
plt.xlabel('Time')
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.title('Phase space', fontsize=16)
plt.legend()
plt.show()

import pdb; pdb.set_trace()