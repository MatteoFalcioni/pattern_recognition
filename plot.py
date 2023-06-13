import matplotlib.pyplot as plt
import numpy as np

print('please, enter the RNN model of which you want to plot the results. Choices are between: RNN, LSTM, GRU')
model_choice = input().upper()
while model_choice != 'GRU' and model_choice != 'LSTM' and model_choice != 'RNN':
    print(f'model {model_choice} is an invalid keyword. Choose between RNN, LSTM or GRU')
    model_choice = input()
file_toplot = f'toplot/{model_choice}_toplot'

# Read the values from the file
epoch_tr_losses = []
epoch_ev_losses = []
perplexities = []

with open(file_toplot, 'r') as file:
    for line in file:
        tr_loss, ev_loss, perp = line.strip().split()
        epoch_tr_losses.append(float(tr_loss))
        epoch_ev_losses.append(float(ev_loss))
        perplexities.append(float(perp))

NUM_EPOCHS = len(epoch_tr_losses)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(epoch_tr_losses, color='blue', label='training loss')
ax1.plot(epoch_ev_losses, color='orange', label='validation loss')
ax1.set_title(f'Training loss vs evaluation loss over {NUM_EPOCHS} epochs')
ax2.set_title(f'Perplexity over {NUM_EPOCHS} epochs')
ax2.plot(perplexities, color='red', label='perplexity')
plt.subplots_adjust(wspace=0.3)
plt.legend()
plt.show()

data = np.genfromtxt('efficiency.txt', delimiter='\t', skip_header=1)
data_to_plot = data[:, 1:]

num_columns = data_to_plot.shape[1]
# Create the x-axis values
x = np.arange(num_columns)
# Create the bar plot
plt.bar(x, data_to_plot, align='center')
# Add x-axis tick labels
plt.xticks(x, ['hamming\ndistance', 'cosine\nsimilarity', 'accuracy'])

# Add labels and title
plt.xlabel('Measures')
plt.ylabel('Values')
plt.title(f'Bar Plot of the {data[0]} efficiency')

# Display the plot
plt.show()




