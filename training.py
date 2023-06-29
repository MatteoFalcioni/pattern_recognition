import torch
import torch.nn as nn
import math

"""
this file contains the functions to train and validate the chosen model, and to eventually save its data
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


def validate(model, ev_dataloader, loss_fn=nn.CrossEntropyLoss()):
    """
        this function validates the given model on the validation data, returning the loss for each epoch
            Parameters:
                model: the model to validate
                ev_dataloader: the dataloader to get the validation data
                loss_fn: the loss function to compute the loss
            Returns:
                the validation loss and the perplexity (both averaged over validation steps)
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


def train_epochs(model, tr_dataloader, ev_dataloader, num_epochs, min_epochs, optimizer, scheduler, loss_fn=nn.CrossEntropyLoss()):
    """
        this function trains the chosen model over epochs, while also checking for overfitting
            Parameters:
                    model: the chosen model between RNN, LSTM and GRU
                    tr_dataloader: the dataloader to get the training data
                    ev_dataloader: the dataloader to get the validation data
                    num_epochs: the number of epochs for which the model will be trained
                    min_epochs: the epoch number from which the checking for overfitting will start
                    optimizer: the chosen optimizer for the loss
                    scheduler: the chosen scheduler for lr decay
                    loss_fn: the loss function to compute the loss
            Returns:
                lists containing training loss, validation loss and perplexity for each epoch
        """

    previous_val_loss = float('inf')
    epoch_tr_losses = []
    epoch_ev_losses = []
    epoch_perplexities = []

    print('starting training and evaluation...')
    for epoch in range(num_epochs):
        print(f'epoch [{epoch}/{num_epochs}]')

        # training
        epoch_tr_loss = train(model, tr_dataloader, optimizer, loss_fn)
        epoch_tr_losses.append(epoch_tr_loss)

        # validation
        epoch_ev_loss, epoch_perplexity = validate(model, ev_dataloader, loss_fn)
        epoch_ev_losses.append(epoch_ev_loss)
        epoch_perplexities.append(epoch_perplexity)

        # Check for overfitting
        if epoch_ev_loss >= previous_val_loss and epoch > min_epochs:
            print("Overfitting detected! Stopping the training loop...")
            break
        # Update the previous validation loss variable
        previous_val_loss = epoch_ev_loss

        scheduler.step()  # lr = lr*DECAY_RATE after DECAY_STEP steps

        print(f'avg epoch #{epoch} train loss: {epoch_tr_losses[epoch]}\navg epoch'
              f' #{epoch} validation loss: {epoch_ev_losses[epoch]}')

    return epoch_tr_losses, epoch_ev_losses, epoch_perplexities


def save_data(model_choice, epoch_tr_losses, epoch_ev_losses, epoch_perplexities, training_time, save, model):
    """
    this function writes training data to file for future plotting
        Parameters:
                model_choice: the chosen model between RNN, LSTM and GRU
                epoch_tr_losses: the training loss over epochs
                epoch_ev_losses: the validation loss over epochs
                epoch_perplexities: the perplexity over epochs
                training_time: the elapsed time during training
                save: string values between [saving, discard] to choose whether to save the model's state_dict() or not
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


def inference(model, seed_seq='nel mezzo del cammin di nostra vita', sample_len=250):
    """
    this function performs inference from the chosen model, generating text from a seed sequence
        Parameters:
            model: the chosen model
            seed_seq: the chosen sequence which the model has to continue with thext generation
            sample_len: the lenght of the sampled text
        Returns:
            the sampled text from the model
    """
    sample_seq = [c for c in seed_seq]
    for step in range(sample_len):
        prediction = model.sample(sample_seq)
        sample_seq.append(prediction)
    sampled_txt = ''.join(sample_seq)

    return sampled_txt

