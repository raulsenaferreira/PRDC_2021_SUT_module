import os
import numpy as np
import pickle
import psutil
from src.utils import util
import matplotlib.pyplot as plt



def plot_images(title, data, labels, similarities, num_row, num_col):

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        try:
            ax = axes[i//num_col, i%num_col]
            ax.imshow(np.squeeze(data[i]), cmap='gray')
            ax.set_title('{}-Sim={}'.format(labels[i], similarities[i]))
            ax.set_axis_off()
        except Exception as e:
            pass    
    fig.suptitle(title)    
    plt.tight_layout(pad=3.0)
    plt.show()


def run(X_test, y_test, experiment, dataset_name):
    arrPred = []
    
    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))
    
    model = experiment.model

    #memory
    process = psutil.Process(os.getpid())

    for img, lbl in zip(X_test, y_test):
                
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log

        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
            
    memory = process.memory_info().rss / 1024 / 1024

    return arrPred, y_test, memory