import torch
import torch.nn as nn
import torch.nn.functional as F
from .spikeLayer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG16_BN(nn.Module):
    def __init__(self):
        super(VGG16_BN, self).__init__()
        # GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN1_1 = nn.BatchNorm2d(num_features=64)

        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN1_2 = nn.BatchNorm2d(num_features=64)

        self.maxpool1 = nn.AvgPool2d(2)
        # GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN2_1 = nn.BatchNorm2d(num_features=128)

        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN2_2 = nn.BatchNorm2d(num_features=128)

        self.maxpool2 = nn.AvgPool2d(2)
        # GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN3_1 = nn.BatchNorm2d(num_features=256)

        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN3_2 = nn.BatchNorm2d(num_features=256)

        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1,
                                 bias=True)

        self.BN3_3 = nn.BatchNorm2d(num_features=256)

        self.maxpool3 = nn.AvgPool2d(2)
        # GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN4_1 = nn.BatchNorm2d(num_features=512)

        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN4_2 = nn.BatchNorm2d(num_features=512)

        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1,
                                 bias=True)

        self.BN4_3 = nn.BatchNorm2d(num_features=512)

        self.maxpool4 = nn.AvgPool2d(2)
        # GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN5_1 = nn.BatchNorm2d(num_features=512)

        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN5_2 = nn.BatchNorm2d(num_features=512)

        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1,
                                 bias=True)

        self.BN5_3 = nn.BatchNorm2d(num_features=512)

        self.maxpool5 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(in_features=512 * 1 * 1, out_features=4096, bias=True)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)

        self.fc3 = nn.Linear(in_features=4096, out_features=10, bias=True)

        self.relu = F.relu

    def forward(self, x, epoch):
        # GROUP 1
        output = self.conv1_1(x)
        output = self.BN1_1(output)
        output = self.relu(output)

        output = self.conv1_2(output)
        output = self.BN1_2(output)
        output = self.relu(output)

        output = self.maxpool1(output)
        # GROUP 2
        output = self.conv2_1(output)
        output = self.BN2_1(output)
        output = self.relu(output)

        output = self.conv2_2(output)
        output = self.BN2_2(output)
        output = self.relu(output)

        output = self.maxpool2(output)
        # GROUP 3
        output = self.conv3_1(output)
        output = self.BN3_1(output)
        output = self.relu(output)

        output = self.conv3_2(output)
        output = self.BN3_2(output)
        output = self.relu(output)

        output = self.conv3_3(output)
        output = self.BN3_3(output)
        output = self.relu(output)

        output = self.maxpool3(output)
        # GROUP 4

        output = self.conv4_1(output)
        output = self.BN4_1(output)
        output = self.relu(output)

        output = self.conv4_2(output)
        output = self.BN4_2(output)
        output = self.relu(output)

        output = self.conv4_3(output)
        output = self.BN4_3(output)
        output = self.relu(output)

        output = self.maxpool4(output)
        # GROUP 5
        output = self.conv5_1(output)
        output = self.BN5_1(output)
        output = self.relu(output)

        output = self.conv5_2(output)
        output = self.BN5_2(output)
        output = self.relu(output)

        output = self.conv5_3(output)
        output = self.BN5_3(output)
        output = self.relu(output)

        output = self.maxpool5(output)

        output = output.view(x.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


class VGG16_BN_optimalThres(nn.Module):
    def __init__(self):
        super().__init__()
        # GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN1_1 = nn.BatchNorm2d(num_features=64)

        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN1_2 = nn.BatchNorm2d(num_features=64)

        self.maxpool1 = nn.AvgPool2d(2)
        # GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN2_1 = nn.BatchNorm2d(num_features=128)

        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN2_2 = nn.BatchNorm2d(num_features=128)

        self.maxpool2 = nn.AvgPool2d(2)
        # GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN3_1 = nn.BatchNorm2d(num_features=256)

        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1), bias=True)
        self.BN3_2 = nn.BatchNorm2d(num_features=256)

        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1,
                                 bias=True)

        self.BN3_3 = nn.BatchNorm2d(num_features=256)

        self.maxpool3 = nn.AvgPool2d(2)
        # GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN4_1 = nn.BatchNorm2d(num_features=512)

        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN4_2 = nn.BatchNorm2d(num_features=512)

        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1,
                                 bias=True)
        self.BN4_3 = nn.BatchNorm2d(num_features=512)

        self.maxpool4 = nn.AvgPool2d(2)
        # GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN5_1 = nn.BatchNorm2d(num_features=512)

        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.BN5_2 = nn.BatchNorm2d(num_features=512)

        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1,
                                 bias=True)
        self.BN5_3 = nn.BatchNorm2d(num_features=512)

        self.maxpool5 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(in_features=512 * 1 * 1, out_features=4096, bias=True)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)

        self.fc3 = nn.Linear(in_features=4096, out_features=10, bias=True)

        self.relu = F.relu
        self.max_active = [0] * 16

    def init_thresh(self, x):
        # GROUP 1
        output = self.conv1_1(x)
        output = self.BN1_1(output)
        output = self.relu(output)
        self.max_active[0] = torch.zeros_like(output)

        output = self.conv1_2(output)
        output = self.BN1_2(output)
        output = self.relu(output)
        self.max_active[1] = torch.zeros_like(output)

        output = self.maxpool1(output)
        # GROUP 2
        output = self.conv2_1(output)
        output = self.BN2_1(output)
        output = self.relu(output)
        self.max_active[2] = torch.zeros_like(output)

        output = self.conv2_2(output)
        output = self.BN2_2(output)
        output = self.relu(output)
        self.max_active[3] = torch.zeros_like(output)

        output = self.maxpool2(output)
        # GROUP 3
        output = self.conv3_1(output)
        output = self.BN3_1(output)
        output = self.relu(output)
        self.max_active[4] = torch.zeros_like(output)

        output = self.conv3_2(output)
        output = self.BN3_2(output)
        output = self.relu(output)
        self.max_active[5] = torch.zeros_like(output)

        output = self.conv3_3(output)
        output = self.BN3_3(output)
        output = self.relu(output)
        self.max_active[6] = torch.zeros_like(output)

        output = self.maxpool3(output)
        # GROUP 4

        output = self.conv4_1(output)
        output = self.BN4_1(output)
        output = self.relu(output)
        self.max_active[7] = torch.zeros_like(output)

        output = self.conv4_2(output)
        output = self.BN4_2(output)
        output = self.relu(output)
        self.max_active[8] = torch.zeros_like(output)

        output = self.conv4_3(output)
        output = self.BN4_3(output)
        output = self.relu(output)
        self.max_active[9] = torch.zeros_like(output)

        output = self.maxpool4(output)
        # GROUP 5
        output = self.conv5_1(output)
        output = self.BN5_1(output)
        output = self.relu(output)
        self.max_active[10] = torch.zeros_like(output)

        output = self.conv5_2(output)
        output = self.BN5_2(output)
        output = self.relu(output)
        self.max_active[11] = torch.zeros_like(output)

        output = self.conv5_3(output)
        output = self.BN5_3(output)
        output = self.relu(output)
        self.max_active[12] = torch.zeros_like(output)

        output = self.maxpool5(output)

        output = output.view(x.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        self.max_active[13] = torch.zeros_like(output)
        output = self.fc2(output)
        output = self.relu(output)
        self.max_active[14] = torch.zeros_like(output)
        output = self.fc3(output)
        self.max_active[15] = torch.zeros_like(output)

    def forward(self, x):
        # GROUP 1
        output = self.conv1_1(x)
        output = self.BN1_1(output)
        output = self.relu(output)
        self.max_active[0] = torch.where(self.max_active[0] > output, self.max_active[0], output)

        output = self.conv1_2(output)
        output = self.BN1_2(output)
        output = self.relu(output)
        self.max_active[1] = torch.where(self.max_active[1] > output, self.max_active[1], output)

        output = self.maxpool1(output)
        # GROUP 2
        output = self.conv2_1(output)
        output = self.BN2_1(output)
        output = self.relu(output)
        self.max_active[2] = torch.where(self.max_active[2] > output, self.max_active[2], output)

        output = self.conv2_2(output)
        output = self.BN2_2(output)
        output = self.relu(output)
        self.max_active[3] = torch.where(self.max_active[3] > output, self.max_active[3], output)

        output = self.maxpool2(output)
        # GROUP 3
        output = self.conv3_1(output)
        output = self.BN3_1(output)
        output = self.relu(output)
        self.max_active[4] = torch.where(self.max_active[4] > output, self.max_active[4], output)

        output = self.conv3_2(output)
        output = self.BN3_2(output)
        output = self.relu(output)
        self.max_active[5] = torch.where(self.max_active[5] > output, self.max_active[5], output)

        output = self.conv3_3(output)
        output = self.BN3_3(output)
        output = self.relu(output)
        self.max_active[6] = torch.where(self.max_active[6] > output, self.max_active[6], output)

        output = self.maxpool3(output)
        # GROUP 4

        output = self.conv4_1(output)
        output = self.BN4_1(output)
        output = self.relu(output)
        self.max_active[7] = torch.where(self.max_active[7] > output, self.max_active[7], output)

        output = self.conv4_2(output)
        output = self.BN4_2(output)
        output = self.relu(output)
        self.max_active[8] = torch.where(self.max_active[8] > output, self.max_active[8], output)

        output = self.conv4_3(output)
        output = self.BN4_3(output)
        output = self.relu(output)
        self.max_active[9] = torch.where(self.max_active[9] > output, self.max_active[9], output)

        output = self.maxpool4(output)
        # GROUP 5
        output = self.conv5_1(output)
        output = self.BN5_1(output)
        output = self.relu(output)
        self.max_active[10] = torch.where(self.max_active[10] > output, self.max_active[10], output)

        output = self.conv5_2(output)
        output = self.BN5_2(output)
        output = self.relu(output)
        self.max_active[11] = torch.where(self.max_active[11] > output, self.max_active[11], output)

        output = self.conv5_3(output)
        output = self.BN5_3(output)
        output = self.relu(output)
        self.max_active[12] = torch.where(self.max_active[12] > output, self.max_active[12], output)

        output = self.maxpool5(output)

        output = output.view(x.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        self.max_active[13] = torch.where(self.max_active[13] > output, self.max_active[13], output)
        output = self.fc2(output)
        output = self.relu(output)
        self.max_active[14] = torch.where(self.max_active[14] > output, self.max_active[14], output)
        output = self.fc3(output)
        self.max_active[15] = torch.where(self.max_active[15] > output, self.max_active[15], output)
        return output


class VGG16_BN_PosNeg_spiking(nn.Module):
    def __init__(self, thresh_list, model):
        super().__init__()
        # group1

        self.conv1_1 = SPIKE_PosNeg_layer_BN(thresh_list[0], -thresh_list[0], model.conv1_1, model.BN1_1)

        self.conv1_2 = SPIKE_PosNeg_layer_BN(thresh_list[1], -thresh_list[1], model.conv1_2, model.BN1_2)

        self.pool1 = nn.AvgPool2d(2)
        # group2

        self.conv2_1 = SPIKE_PosNeg_layer_BN(thresh_list[2], -thresh_list[2], model.conv2_1, model.BN2_1)

        self.conv2_2 = SPIKE_PosNeg_layer_BN(thresh_list[3], -thresh_list[3], model.conv2_2, model.BN2_2)

        self.pool2 = nn.AvgPool2d(2)
        # group3

        self.conv3_1 = SPIKE_PosNeg_layer_BN(thresh_list[4], -thresh_list[4], model.conv3_1, model.BN3_1)

        self.conv3_2 = SPIKE_PosNeg_layer_BN(thresh_list[5], -thresh_list[5], model.conv3_2, model.BN3_2)

        self.conv3_3 = SPIKE_PosNeg_layer_BN(thresh_list[6], -thresh_list[6], model.conv3_3, model.BN3_3)

        self.pool3 = nn.AvgPool2d(2)
        # group4

        self.conv4_1 = SPIKE_PosNeg_layer_BN(thresh_list[7], -thresh_list[7], model.conv4_1, model.BN4_1)

        self.conv4_2 = SPIKE_PosNeg_layer_BN(thresh_list[8], -thresh_list[8], model.conv4_2, model.BN4_2)

        self.conv4_3 = SPIKE_PosNeg_layer_BN(thresh_list[9], -thresh_list[9], model.conv4_3, model.BN4_3)

        self.pool4 = nn.AvgPool2d(2)
        # group5

        self.conv5_1 = SPIKE_PosNeg_layer_BN(thresh_list[10], -thresh_list[10], model.conv5_1, model.BN5_1)

        self.conv5_2 = SPIKE_PosNeg_layer_BN(thresh_list[11], -thresh_list[11], model.conv5_2, model.BN5_2)

        self.conv5_3 = SPIKE_PosNeg_layer_BN(thresh_list[12], -thresh_list[12], model.conv5_3, model.BN5_3)

        self.pool5 = nn.AvgPool2d(2)

        self.fc1 = SPIKE_PosNeg_layer(thresh_list[13], -thresh_list[13], model.fc1)
        self.fc2 = SPIKE_PosNeg_layer(thresh_list[14], -thresh_list[14], model.fc2)
        self.fc3 = SPIKE_PosNeg_layer(thresh_list[15], -thresh_list[15], model.fc3)
        self.T = 32

    def weight_bias_norm(self):
        self.conv1_1.compute_Conv_weight()
        self.conv1_2.compute_Conv_weight()
        self.conv2_1.compute_Conv_weight()
        self.conv2_2.compute_Conv_weight()
        self.conv3_1.compute_Conv_weight()
        self.conv3_2.compute_Conv_weight()
        self.conv3_3.compute_Conv_weight()
        self.conv4_1.compute_Conv_weight()
        self.conv4_2.compute_Conv_weight()
        self.conv4_3.compute_Conv_weight()
        self.conv5_1.compute_Conv_weight()
        self.conv5_2.compute_Conv_weight()
        self.conv5_3.compute_Conv_weight()

    def init_layer(self):
        self.conv1_1.init_mem()
        self.conv1_2.init_mem()
        self.conv2_1.init_mem()
        self.conv2_2.init_mem()
        self.conv3_1.init_mem()
        self.conv3_2.init_mem()
        self.conv3_3.init_mem()
        self.conv4_1.init_mem()
        self.conv4_2.init_mem()
        self.conv4_3.init_mem()
        self.conv5_1.init_mem()
        self.conv5_2.init_mem()
        self.conv5_3.init_mem()
        self.fc1.init_mem()
        self.fc2.init_mem()
        self.fc3.init_mem()

    def forward(self, x):
        self.init_layer()
        with torch.no_grad():
            out_spike_sum = 0
            for time in range(self.T):
                spike_input = x
                output, m1_1 = self.conv1_1(spike_input, time)
                output, m1_2 = self.conv1_2(output, time)
                output = self.pool1(output)
                # group 2
                output, m2_1 = self.conv2_1(output, time)
                output, m2_2 = self.conv2_2(output, time)
                output = self.pool2(output)
                # group 3
                output, m3_1 = self.conv3_1(output, time)
                output, m3_2 = self.conv3_2(output, time)
                output, m3_3 = self.conv3_3(output, time)
                output = self.pool3(output)
                # group 4
                output, m4_1 = self.conv4_1(output, time)
                output, m4_2 = self.conv4_2(output, time)
                output, m4_3 = self.conv4_3(output, time)
                output = self.pool4(output)
                # group 5
                output, m5_1 = self.conv5_1(output, time)
                output, m5_2 = self.conv5_2(output, time)
                output, m5_3 = self.conv5_3(output, time)
                output = self.pool5(output)
                #
                output = output.view(output.size(0), -1)
                output, mfc1 = self.fc1(output, time)
                output, mfc2 = self.fc2(output, time)
                output, mfc3 = self.fc3(output, time)

                out_spike_sum += output
                if (time + 1) == self.T:
                    sub_result = out_spike_sum / (time + 1)

        return sub_result
