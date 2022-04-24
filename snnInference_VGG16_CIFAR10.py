import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from models.VGG import *
from tqdm import tqdm

T = 32
print(T)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

model_save_name = './MyWeights/VGG16_CIFAR10_state_dict-bn-.pth'
data_path = './raw/'  # dataset path


def dataset():
    trans_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                                                 transform=trans_train)
    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True,
                                            transform=trans_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=True)
    return train_loader, test_loader


train_loader, test_loader = dataset()

ann = VGG16_BN_optimalThres()

ann.to(device)
if torch.cuda.is_available():
    pretrained_dict = torch.load(model_save_name, map_location=torch.device('cuda'))
else:
    pretrained_dict = torch.load(model_save_name, map_location=torch.device('cpu'))

ann.load_state_dict(pretrained_dict)

## # validate on test set with ANN.
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        ann.eval()
        inputs = inputs.to(device)
        targets.to(device)
        if batch_idx == 0:
            ann.init_thresh(inputs)
        outputs = ann(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

# # find the maximum activation and min activation on training set
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ann.eval()
        inputs = inputs.to(device)
        targets.to(device)
        outputs = ann(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

print('Test Accuracy of the model on the 50000 train images: %.3f' % (100 * correct / total))

max_activate = ann.max_active

snn = VGG16_BN_PosNeg_spiking(max_activate, ann)
snn.to(device)

total = 0
testCorr = 0
with torch.no_grad():
    snn.weight_bias_norm()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):

        snn.init_layer()
        inputs = inputs.to(device)

        outputs_ls = snn(inputs)
        targets.to(device)
        total += float(targets.cpu().size(0))
        _, predicted = outputs_ls.cpu().max(1)
        testCorr += float(predicted.eq(targets).sum().item())

    corr = 100.000 * testCorr / total

    print('simulation length: ', T, ' -> corr: ', corr)
