import torch
import torch.nn as nn
import math

"""
this file contains the training() and validation() functions to run in the training loop.
"""


def train(model, tr_dataloader, optimizer, loss_fn=nn.CrossEntropyLoss()):
    """
    this function trains the given model on the training data, returning the loss for each epoch
        Parameters:
            model: the model to train
            tr_dataloader: the dataloader to get the training data
            loss_fn: the loss function to compute the loss
            optimizer: the chosen optimizer
        Returns:
            the training loss (averaged over training steps)
    """
    epoch_tr_loss = 0
    n_train = len(tr_dataloader)

    for X, y in tr_dataloader:
        # training
        model.train()

        # forward pass
        outputs, h_n = model(X)
        tr_loss = loss_fn(outputs, y)

        # backward and optimize
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        epoch_tr_loss += tr_loss.item() / float(n_train)

    return epoch_tr_loss


def validation(model, ev_dataloader, loss_fn=nn.CrossEntropyLoss()):
    """
        this function trains the given model on the training data, returning the loss for each epoch
            Parameters:
                model: the model to train
                ev_dataloader: the dataloader to get the validation data
                loss_fn: the loss function to compute the loss
            Returns:
                the validation loss and the perplexity (both averaged over training steps)
    """

    epoch_ev_loss = 0
    epoch_perplexity = 0

    n_eval = len(ev_dataloader)
    for X, y in ev_dataloader:
        # evaluation
        model.eval()

        with torch.no_grad():
            # evaluate loss on eval set
            out, _ = model(X)
            ev_loss = loss_fn(out, y)

            epoch_ev_loss += ev_loss.item() / float(n_eval)
            epoch_perplexity += math.exp(ev_loss.item()) / float(n_eval)  # 'average' perplexity

    return epoch_ev_loss, epoch_perplexity


def save_data(model_choice, epoch_tr_losses, epoch_ev_losses, epoch_perplexities, training_time, save, model):
    """
    this function writes training data to file for future plotting
        Parameters:
                model_choice: the chosen model between RNN, LSTM and GRU
                epoch_tr_losses: the training loss over epochs
                epoch_ev_losses: the validation loss over epochs
                epoch_perplexities: the perplexity over epochs
                training_time: the elapsed time during training
                SAVE: string values between [saving, discard] to choose whether to save the model's state_dict or not
                model: the model whose state_dict should be saved or discarded
        Returns:
            None
    """
    # write losses to file
    file_toplot = f'toplot/{model_choice}_toplot.txt'
    with open(file_toplot, 'w') as file:
        # Zip the lists and iterate over the pairs
        for tr_loss, ev_loss, perplexity in zip(epoch_tr_losses, epoch_ev_losses, epoch_perplexities):
            # Write the values to the file with a space in between
            file.write(f'{tr_loss}\t{ev_loss}\t{perplexity}\n')

    with open('toplot/efficiency.txt', 'a') as file:    # write training time to file
        file.write(f'\n{model_choice}\t{training_time}')

    if save == 'saving':
        # saving the model's state_dict
        path = f'pretrained/{model_choice}.pth'
        torch.save(model.state_dict(), path)  # this is saved in train mode (!)
        print('model has been saved successfully')

