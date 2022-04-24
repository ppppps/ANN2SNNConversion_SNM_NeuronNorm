import torch
import torchvision
import torchvision.transforms as transforms
import os
import time

from models.VGG import *
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
learning_rate = 1e-2
num_epochs = 200
data_path = './raw/'

model_save_name = './MyWeights/VGG16_CIFAR10_state_dict-bn-.pth'

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



if __name__ == "__main__":

    best_acc = 0
    best_epoch = 0
    ann = VGG16_BN()

    ann.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(ann.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200, 240], gamma=0.2)

    train_loader, test_loader = dataset()
    for epoch in tqdm(range(num_epochs)):

        correctTrain = 0
        totalTrain = 0
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            ann.train()
            ann.zero_grad()
            optimizer.zero_grad()
            labels.to(device)
            images = images.float().to(device)
            outputs = ann(images, epoch)

            _, predicted = outputs.cpu().max(1)
            totalTrain += float(labels.size(0))
            correctTrain += float(predicted.eq(labels).sum().item())

            loss = criterion(outputs.cpu(), labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('Test Accuracy of the model on the 50000 train images: %.3f' % (100 * correctTrain / totalTrain))

        correctTest = 0
        totalTest = 0
        scheduler.step()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                ann.eval()
                inputs = inputs.to(device)
                optimizer.zero_grad()
                targets.to(device)
                outputs = ann(inputs, epoch)
                loss = criterion(outputs.cpu(), targets)
                _, predicted = outputs.cpu().max(1)
                totalTest += float(targets.size(0))
                correctTest += float(predicted.eq(targets).sum().item())

        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correctTest / totalTest))

        accTest = 100. * float(correctTest) / float(totalTest)
        if best_acc < accTest:
            best_acc = accTest
            best_epoch = epoch
            torch.save(ann.state_dict(), model_save_name)

        print('best_acc is: ', best_acc, ' find in epoch: ', best_epoch)
        print('Iters:', epoch, '\n\n\n')
