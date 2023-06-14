import matplotlib.pyplot as plt
import numpy as np

RNN_toplot = 'toplot/RNN_toplot.txt'
LSTM_toplot = 'toplot/LSTM_toplot.txt'
GRU_toplot = 'toplot/GRU_toplot.txt'

# Read data from text files and store in separate lists
with open('toplot/RNN_toplot.txt') as file:
    data_model1 = [line.split() for line in file]
with open('toplot/LSTM_toplot.txt') as file:
    data_model2 = [line.split() for line in file]
with open('toplot/GRU_toplot.txt') as file:
    data_model3 = [line.split() for line in file]

train_loss_model1 = [float(row[0]) for row in data_model1]
eval_loss_model1 = [float(row[1]) for row in data_model1]
perplexity_model1 = [float(row[2]) for row in data_model1]

train_loss_model2 = [float(row[0]) for row in data_model2]
eval_loss_model2 = [float(row[1]) for row in data_model2]
perplexity_model2 = [float(row[2]) for row in data_model2]

train_loss_model3 = [float(row[0]) for row in data_model3]
eval_loss_model3 = [float(row[1]) for row in data_model3]
perplexity_model3 = [float(row[2]) for row in data_model3]

RNN_epochs = len(train_loss_model1)
LSTM_epochs = len(train_loss_model2)
GRU_epochs = len(train_loss_model3)

# Create a 2x3 subplot figure
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Plot training loss and evaluation loss on the first row
# RNN
axs[0, 0].plot(train_loss_model1, label='RNN Train Loss')
axs[0, 0].plot(eval_loss_model1, label='RNN Eval Loss')
axs[0, 0].set_title('RNN Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
# Plot perplexity on the second row
axs[1, 0].plot(perplexity_model1, label='RNN Perplexity', color='red')
axs[1, 0].set_title('RNN Perplexity')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Perplexity')
axs[1, 0].legend()

# LSTM
axs[0, 1].plot(train_loss_model2, label='LSTM Train Loss')
axs[0, 1].plot(eval_loss_model2, label='LSTM Eval Loss')
axs[0, 1].set_title('LSTM Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[1, 1].plot(perplexity_model2, label='LSTM Perplexity', color='red')
axs[1, 1].set_title('LSTM Perplexity')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Perplexity')
axs[1, 1].legend()

# GRU
axs[0, 2].plot(train_loss_model3, label='GRU Train Loss')
axs[0, 2].plot(eval_loss_model3, label='GRU Eval Loss')
axs[0, 2].set_title('GRU Loss')
axs[0, 2].set_xlabel('Epochs')
axs[0, 2].set_ylabel('Loss')
axs[0, 2].legend()
axs[1, 2].plot(perplexity_model3, label='GRU Perplexity', color='red')
axs[1, 2].set_title('GRU Perplexity')
axs[1, 2].set_xlabel('Epochs')
axs[1, 2].set_ylabel('Perplexity')
axs[1, 2].legend()

# Customize tick parameters on shared y-axis
for ax in axs[:, 1]:
    ax.tick_params(left=True, labelleft=True)
for ax in axs[:, 2]:
    ax.tick_params(left=True, labelleft=True)

# Adjust spacing between subplots
plt.tight_layout()
# Display the plot
plt.show()

# -----------------------------

# Read data from the file and store in separate lists
with open('toplot/efficiency.txt') as file:
    data = [line.split() for line in file]

labels = [row[0] for row in data]
values = [float(row[1]) for row in data]

# Set the width of the bars
bar_width = 0.2
# Set the positions of the bars on the x-axis
x_pos = np.arange(len(labels))

# Plot the bars
plt.bar(labels, values, width=bar_width, label='training time')

# Set the x-axis tick labels to be the labels from the first column
plt.xticks(x_pos, labels)

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Elapsed time (seconds)')
plt.title('Elapsed time during the training process')

# Add legend
plt.legend()

# Show the plot
plt.show()




