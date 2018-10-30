import tensorflow as tf
import numpy as np
import random
import math
import os
import xlwt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os import path
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

graph_x = range(1001)
layer_num = 6
loop_count = 0

ax = plt.subplot()
dataset_flag = 2
if dataset_flag == 0:
    proposal = np.load("./sotsuronData_1000_500_500/proposal/proposal_mnist.npy")
    data_SAE = np.load("./sotsuronData_1000_500_500/SAE/SAE_mnist.npy")
    data_GSD = np.load("./sotsuronData_1000_500_500/GSD/GSD_mnist.npy")
    data_GSD_last_tune = np.load("GSD_last_tune_mnist.npy")
    data_MLP = np.load("./sotsuronData_1000_500_500/MLP/MLP_mnist.npy")
    data_GFNN = np.load("GFNN_mnist.npy")
if dataset_flag == 1:
    proposal = np.load("./sotsuronData_1000_500_500/proposal/proposal_cifar10.npy")
    data_SAE = np.load("./sotsuronData_1000_500_500/SAE/SAE_cifar10.npy")
    data_GSD = np.load("./sotsuronData_1000_500_500/GSD/GSD_cifar10.npy")
    data_GSD_last_tune = np.load("GSD_last_tune_cifar10.npy")
    data_MLP = np.load("./sotsuronData_1000_500_500/MLP/MLP_cifar10.npy")
    data_GFNN = np.load("GFNN_cifar10.npy")
if dataset_flag == 2:
    proposal = np.load("./sotsuronData_1000_500_500/proposal/proposal_cifar100.npy")
    data_SAE = np.load("./sotsuronData_1000_500_500/SAE/SAE_cifar100.npy")
    data_GSD = np.load("./sotsuronData_1000_500_500/GSD/GSD_cifar100.npy")
    data_GSD_last_tune = np.load("GSD_last_tune_cifar100.npy")
    data_MLP = np.load("./sotsuronData_1000_500_500/MLP/MLP_cifar100.npy")
    data_GFNN = np.load("GFNN_cifar100.npy")

graph_tmp_proposal = [None for i in range(1000)]
for i in range(len(proposal)):
    for j in range(len(proposal[i])):
        if proposal[i][j] is not None:
            graph_tmp_proposal[j] = proposal[i][j]
            if j > loop_count:
                loop_count = j
graph_tmp_proposal = np.append([0], graph_tmp_proposal)

print(data_SAE.shape)
first_not_zero_counter = 0
for j in range(len(data_SAE)):
    if data_SAE[j] is not None:
        if first_not_zero_counter == 0:
            first_not_zero_counter = 1
            for k in range(j):
                data_SAE[k] = 0
        if j > loop_count:
            loop_count = j
data_SAE = np.append([0], data_SAE)

print(data_GSD.shape)
graph_tmp_GSD = [None for i in range(1000)]
"""
for i in range(len(data_GSD)):
    if data_GSD[i] is not None:
        graph_tmp_GSD[i] = data_GSD[i]
"""
for i in range(len(data_GSD)):
    for j in range(len(data_GSD[i])):
        if data_GSD[i][j] is not None:
            graph_tmp_GSD[j] = data_GSD[i][j]
        if i > loop_count:
            loop_count = i
graph_tmp_GSD = np.append([0], graph_tmp_GSD)

graph_tmp_GSD_last_tune = [None for i in range(1000)]
"""
for i in range(len(data_GSD)):
    if data_GSD[i] is not None:
        graph_tmp_GSD[i] = data_GSD[i]
"""
for i in range(len(data_GSD_last_tune)):
    for j in range(len(data_GSD_last_tune[i])):
        if data_GSD_last_tune[i][j] is not None:
            graph_tmp_GSD_last_tune[j] = data_GSD_last_tune[i][j]
        if i > loop_count:
            loop_count = i
graph_tmp_GSD_last_tune = np.append([0], graph_tmp_GSD_last_tune)

print(data_MLP.shape)
graph_tmp_MLP = [None for i in range(1000)]
for j in range(len(data_MLP)):
    if data_MLP[j] is not None:
        graph_tmp_MLP[j] = data_MLP[j]
        if j > loop_count:
            loop_count = j
graph_tmp_MLP = np.append([0], graph_tmp_MLP)

print(data_GFNN.shape)
graph_tmp_GFNN = [None for i in range(1000)]
for j in range(len(data_MLP)):
    if data_GFNN[j] is not None:
        graph_tmp_GFNN[j] = data_GFNN[j]
        if j > loop_count:
            loop_count = j
graph_tmp_GFNN = np.append([0], graph_tmp_GFNN)

ax.plot(graph_x, graph_tmp_proposal, '-', linewidth=1.0, markersize=10, label="proposed", color="Red", antialiased=True)
ax.plot(graph_x, graph_tmp_GSD, '--', linewidth=1.0, markersize=10, label="GSD", color="Green", antialiased=True)
ax.plot(graph_x, data_SAE, '-.', linewidth=1.0, markersize=10, label="SAE", color="Blue", antialiased=True)
ax.plot(graph_x, graph_tmp_MLP, ':', linewidth=1.0, markersize=10, label="MLP", color="Black", antialiased=True)

# ax.plot(graph_x, graph_tmp_GFNN, linewidth=1.0, markersize=10, label="GF-MLP", color="orange", antialiased=True)
# ax.plot(graph_x, graph_tmp_GSD_last_tune, linewidth=1.0, markersize=10, label="GSD*", color="blueviolet", antialiased=True)

ax.grid(which="both")
ax.set_xlabel("epoch", fontsize=15)
ax.set_ylabel("accuracy", fontsize=15)
ax.set_xlim([0, loop_count + 1])
plt.legend(loc='lower right', fontsize=15)
plt.tick_params(labelsize=15)
plt.show()

ax = plt.subplot()
proposal0 = np.load("proposal_y.npy")[0]
proposal1 = np.load("proposal_y.npy")[1]
graph_tmp_proposal0 = [None for i in range(1000)]
graph_tmp_proposal1 = [None for i in range(1000)]
graph_tmp_proposal_inner = [None for i in range(1000)]
last = 0
for i in range(len(proposal0)):
    if proposal0[i] is not None:
        graph_tmp_proposal0[i] = proposal0[i]
        last = i
graph_tmp_proposal_inner[last] = proposal0[last]
first_flag = 1
for i in range(len(proposal1)):
    if proposal1[i] is not None:
        if first_flag == 1:
            graph_tmp_proposal_inner[i] = proposal1[i]
            first_flag = 0
        graph_tmp_proposal1[i] = proposal1[i]

graph_tmp_proposal0 = np.append([0], graph_tmp_proposal0)
graph_tmp_proposal1 = np.append([None], graph_tmp_proposal1)
graph_tmp_proposal_inner = np.append([None], graph_tmp_proposal_inner)

ax.plot(graph_x, graph_tmp_proposal_inner, '-', linewidth=1.0, antialiased=True, color="Red")#0  # "#ffa500ff")
ax.plot(graph_x, graph_tmp_proposal1, linewidth=1.0, marker=".", markersize=10,
        label="3 hidden layers", antialiased=True, color="Red")#0
ax.plot(graph_x, graph_tmp_proposal0,dashes=[6, 6], #'--',
        linewidth=1.0, marker=".", markersize=10,
        label="2 hidden layers", antialiased=True, color="Red")#1
ax.grid(which="both")
ax.set_xlabel("epoch", fontsize=15)
ax.set_ylabel("accuracy", fontsize=15)
ax.set_xlim([0, 14])
ax.set_ylim([0.3, 0.57])
plt.legend(loc='lower right', fontsize=15)
plt.tick_params(labelsize=15)
plt.xticks(np.arange(0, 20.0, 5))
plt.show()
