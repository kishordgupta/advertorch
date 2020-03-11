#pip install advertorch

import matplotlib.pyplot as plt
%matplotlib inline

import os
import argparse
import torch
import torch.nn as nn

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"
#torch.load with map_location=torch.device('cpu')
model = LeNet5()
model.load_state_dict(
    torch.load(os.path.join( filename),map_location=torch.device('cpu')))
model.to(device)
model.eval()

batch_size = 10
loader = get_mnist_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)

from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

bits_squeezing = BitSqueezing(bit_depth=5)
median_filter = MedianSmoothing2D(kernel_size=3)
jpeg_filter = JPEGFilter(10)

defense = nn.Sequential(
    jpeg_filter,
    bits_squeezing,
    median_filter,
)
from advertorch.attacks import LBFGSAttack
from advertorch.bpda import BPDAWrapper
defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
defended_model = nn.Sequential(defense_withbpda, model)
bpda_adversary = LBFGSAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),num_classes=10,
    targeted=False)


bpda_adv = bpda_adversary.perturb(cln_data, true_label)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(bpda_adv[ii])
    plt.imsave('./lbfgs/2'+str(ii)+'.png',bpda_adv[ii].squeeze())

bpda_adversary = LBFGSAttack(
    defended_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),num_classes=10,
    targeted=False)


bpda_adv = bpda_adversary.perturb(cln_data, true_label)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(bpda_adv[ii])
    plt.imsave('./lbfgs/2'+str(ii)+'.png',bpda_adv[ii].squeeze())

