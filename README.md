# RNN Implementation in PyTorch
## Comparing Efficiency: RNN vs LSTM vs GRU
### Generating Text based on Dante's Divina Commedia
This repository contains the implementation of a Recurrent Neural Network (RNN) using PyTorch. The main goal of this project is to compare the efficiency of different RNN variants, namely LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit), in generating text based on Dante's Divina Commedia.

## Table of Contents
- [Introduction](#Introduction)
- [Project structure](#Project_structure)

## Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks that are well-suited for sequential data, such as text. They have the ability to retain information from previous steps and use it to make predictions or generate new sequences. In this project, we explore the effectiveness of RNNs by comparing two popular variants: LSTM and GRU.

To evaluate the performance of LSTM and GRU models, we train them on a dataset consisting of Dante's Divina Commedia. The models are then used to generate text that resembles the style and language of the original work.

The architecture of an RNN is characterized by recurrent connections, which form a directed cycle in the network, allowing information to be circulated and preserved across different time steps. These connections enable RNNs to process variable-length input sequences, making them flexible and adaptable to different data modalities. 

To understand the structure of an RNN, let’s consider a basic one-layer RNN with a single “recurrent unit”. At each time step $t$, the network receives an input vector $x(t)$ and produces an output vector $y(t)$. Additionally, the network maintains a hidden state vector $h(t)$, which acts as a memory that encodes information from past time steps. 
The recurrent connection in an RNN is formed by connecting the hidden state from the previous time step $h(t − 1)$ to the current time step $t$. This connection allows the hidden state to influence the computation at the current time step, thus enabling the network to retain information about
past inputs. This process is called the ''unfolding `` of the RNN and is depicted in the image below.

<img src="readme_img/recurrence_unfolding.png" alt="" width="900">

The equations governing the RNN functioning can be summed up as:

<img src="readme_img/RNN_eq.png" alt="" width="300">
<img src="readme_img/RNN_eq2.png" alt="" width="180">

Where $W$ represent the weight matrices for the differrent units, $\mathbf{b}$ and $\mathbf{c}$ are bias terms, $f$ denotes the activation function applied element-wise to the input of the recurrent unit and $g$ is the activation function applied to the output layer.

Unlickily, RNNs suffer from certain limitations (vanishing and exploding gradient issues) that can hinder their performance on tasks that involve long-term dependencies. These limitations motivated the developmentof more advanced architectures like Long Short-Term Memory (LSTM) network and Gated-Recurrent Units (GRU). 

A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. Forget gates decide what information to discard from a previous state. Input gates decide which pieces of new information to store in the current state, using the same system as forget gates. Output gates control which pieces of information in the current state to output. 

In the equations below, matrices $W_{\alpha \beta}$ contain the weights of the input and recurrent connections, where the subscripts $\alpha ,\beta$ will be referring to weights for transitions from gate $\alpha$ to gate $\beta$, which could be input gate $i$, output gate $o$, the forget gate $f$ or the memory cell $c$. In the same way, $b_{\alpha \beta}$ will refer to biases for the layers, and $\sigma_\alpha$ will be the activation functions of the gates.

<img src="readme_img/LSTM_eqs.png" alt="" width="300">

The initial values are $c_0=0$ and $h_0=0$ and the operator $\odot$ denotes the element-wise product. The subscript $t$ indexes the time step.
Here is a sketch of the LSTM architecture:

<img src="readme_img//LSTM.png" alt="" width="400">

Finally, GRUs simplify the LSTM architecture by combining the forget and input gates into a single update gate. Additionally, they introduce a reset gate that determines how much of the previous hidden state should be forgotten. This simplification results in a more streamlined architecture with fewer parameters than LSTMs. The following are its equations: 

<img src="readme_img//GRU_eqs.png" alt="" width="400">

where $r_t, z_t, n_t$ are the reset, update, and new gates, respectively. $\sigma$ is the sigmoid function, and $\odot$ is the Hadamard product.

Here is a sketch of its architecture:

<img src="readme_img//GRU.png" alt="" width="400">

##Project structure


