import numpy as np
import matplotlib.pyplot as plt
import util
from src.Classes.dataset import Dataset


def temporalize_by_class_dist(x, smoothing_steps, distance):
    """
    :param x: An (n_samples, n_dims) dataset
    :return: A (n_samples, ) array of indexes that can be used to shuffle the input for temporal smoothness.
    """
    x_flat = x.reshape(x.shape[0], -1)
    index_buffer = np.arange(1, smoothing_steps+1)
    next_sample_buffer = x_flat[1:smoothing_steps+1].copy()
    shuffling_indices = np.zeros(len(x), dtype=int)
    rectifier = np.abs if distance=='L1' else np.square if distance=='L2' else print("wrong distance metric:", distance)
    
    current_index = 0
    for i in range(len(x)):
        shuffling_indices[i] = current_index
        closest = np.argmin(rectifier(x_flat[current_index]-next_sample_buffer).sum(axis=1))
        current_index = index_buffer[closest]
        weve_aint_done_yet = i+smoothing_steps+1 < len(x)
        next_index = i+smoothing_steps+1
        next_sample_buffer[closest] = x_flat[next_index] if weve_aint_done_yet else float('inf')
        index_buffer[closest] = next_index if weve_aint_done_yet else -1
        
    return shuffling_indices


def demo_temporal_mnist(n_samples = None, smoothing_steps = 200):
    #_, _, original_data, original_labels = get_mnist_dataset(n_training_samples=n_samples, n_test_samples=n_samples).xyxy
    #_, _, _, _, original_data, original_labels, _ = util.load_mnist(onehotencoder=False)
    _, _, _, _, original_data, original_labels, _ = util.load_e_mnist(onehotencoder=False)
    #_, _, temporal_data, temporal_labels = get_temporal_mnist_dataset(n_training_samples=n_samples, n_test_samples=n_samples, smoothing_steps=smoothing_steps).xyxy
    temporal_data, temporal_labels = get_temporal_mnist_dataset(n_training_samples=n_samples, n_test_samples=n_samples, smoothing_steps=smoothing_steps)


def plot_images(temporal_data, temporal_labels, num_row, num_col):

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(np.squeeze(temporal_data[i]), cmap='gray')
        ax.set_title('{}'.format(temporal_labels[i]))
        ax.set_axis_off()
    plt.tight_layout(pad=3.0)
    plt.show()


#testing
'''
#mnist_type=['original', 'rotated', 'extended', 'affnist', 'cvt_mnist', 'cdt_mnist']
mnist_type = ['cht_mnist']
#_, _, _, _, ts_x, ts_y, _ = util.load_mnist(mnist_type='extended', onehotencoder=False)
#plot_images(ts_x, ts_y, num_row=2, num_col=5)

for t in mnist_type:
    x_train, y_train, _, _, x_test, y_test, _ = util.load_mnist(mnist_type=t, onehotencoder=False)
    #tr_ixs = temporalize_by_class_dist(x_train, smoothing_steps=1000, distance='L1')
    #ts_ixs = temporalize_by_class_dist(x_test, smoothing_steps=1000, distance='L1')
    ts_ixs = [i for i in range(len(y_test)) if y_test[i] == 7]
    #print(y_test)
    plot_images(x_test[ts_ixs], y_test[ts_ixs], num_row=5, num_col=9)
'''