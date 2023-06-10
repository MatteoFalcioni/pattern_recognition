# RNN Implementation in PyTorch
## Comparing Efficiency: LSTM vs GRU
### Generating Text based on Dante's Divina Commedia
This repository contains the implementation of a Recurrent Neural Network (RNN) using PyTorch. The main goal of this project is to compare the efficiency of different RNN variants, namely LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit), in generating text based on Dante's Divina Commedia.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

**Introduction**

Recurrent Neural Networks (RNNs) are a class of neural networks that are well-suited for sequential data, such as text. They have the ability to retain information from previous steps and use it to make predictions or generate new sequences. In this project, we explore the effectiveness of RNNs by comparing two popular variants: LSTM and GRU.

To evaluate the performance of LSTM and GRU models, we train them on a dataset consisting of Dante's Divina Commedia. The models are then used to generate text that resembles the style and language of the original work.

The architecture of an RNN is characterized by recurrent connections, which
form a directed cycle in the network, allowing information to be circulated and preserved across
different time steps. These connections enable RNNs to process variable-length input sequences,
making them flexible and adaptable to different data modalities. 

To understand the structure of an RNN, let’s consider a basic one-layer RNN with a single “re-
current unit”. At each time step $t$, the network receives an input vector $x(t)$ and produces an
output vector $y(t)$. Additionally, the network maintains a hidden state vector $h(t)$, which acts as
a memory that encodes information from past time steps. 
The recurrent connection in an RNN is formed by connecting the hidden state from the previous
time step $h(t − 1)$ to the current time step $t$. This connection allows the hidden state to influence
the computation at the current time step, thus enabling the network to retain information about
past inputs. This process is called the ''unfolding `` of the RNN and is depicted in the image below.

![Depiction of the unfolding of an RNN](readme_img/recurrence_unfolding.png)

**Installation**
1. Clone the repository:

git clone https://github.com/yourusername/yourprojectname.git


