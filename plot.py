import matplotlib.pyplot as plt
import numpy as np

# Read the values from the file
epoch_tr_losses = []
epoch_ev_losses = []

with open('toplot.txt', 'r') as file:
    for line in file:
        tr_loss, ev_loss = line.strip().split()
        epoch_tr_losses.append(float(tr_loss))
        epoch_ev_losses.append(float(ev_loss))

NUM_EPOCHS = len(epoch_tr_losses)

plt.figure()
plt.plot(epoch_tr_losses, color='blue', label='training loss')
plt.plot(epoch_ev_losses, color='orange', label='evaluation loss')
plt.title(f'Training loss vs evaluation loss over {NUM_EPOCHS} epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
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




