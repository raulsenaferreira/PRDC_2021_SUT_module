from src.GTRSB_experiments import DNN_outOfBox_dimReduc_test
from src.MNIST_experiments import DNN_outOfBox_dimReduc_MNIST_test
from src.utils import metrics
from time import perf_counter as timer
import numpy as np


def run(repetition, classToMonitor, layer_index, layer_name, models_folder, monitors_folder, isTestOneClass, sep):
    dim_reduc_method = 'isomap'
    avg_acc = ['CNN+OOB+'+dim_reduc_method] #accuracy
    avg_cf = ['CNN+OOB+'+dim_reduc_method] #confusion matrix
    avg_time = ['CNN+OOB+'+dim_reduc_method] #time
    avg_memory = ['CNN+OOB+'+dim_reduc_method] #memory
    avg_F1 = ['CNN+OOB+'+dim_reduc_method] #memory

    acc = []
    t = []
    cf = [[],[],[],[]]
    mem = []
    f1 = []
    datasets = []

    monitor_name = "monitor_Box_"+dim_reduc_method+"_MNIST.p"
    model_name = 'DNN_MNIST.h5'
    
    for i in range(repetition):
        print("MNIST experiment {} of {} ...".format(i+1, repetition))

        ini = timer()
        arrPred, arrLabel, memory, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_dimReduc_MNIST_test.run(
            classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, 
            model_name, isTestOneClass, sep, dim_reduc_method)
        end = timer()

        acc.append(metrics.evaluate(arrLabel, arrPred))
        t.append(end-ini)
        mem.append(memory)
        f1.append(metrics.F1(arrLabel, arrPred))
        cf[0].append(arrFP[str(classToMonitor)])
        cf[1].append(arrFN[str(classToMonitor)])
        cf[2].append(arrTP[str(classToMonitor)])
        cf[3].append(arrTN[str(classToMonitor)])
    
    avg_acc.append(np.mean(acc))
    avg_time.append(np.mean(t))
    avg_memory.append(np.mean(mem))
    avg_F1.append(np.mean(f1))
    avg_cf.append([np.mean(cf[0]), np.mean(cf[1]), np.mean(cf[2]), np.mean(cf[3])])
    datasets.append('MNIST')

    acc = []
    t = []
    cf = [[],[],[],[]]
    mem = []
    f1 = []

    monitor_name = "monitor_Box_"+dim_reduc_method+"_GTRSB.p"
    model_name = 'DNN_GTRSB.h5'

    for i in range(repetition):
        print("GTRSB experiment {} of {} ...".format(i+1, repetition))

        ini = timer()
        arrPred, arrLabel, memory, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_dimReduc_test.run(
            classToMonitor, layer_index, models_folder, monitors_folder, monitor_name, model_name, 
            isTestOneClass, sep, dim_reduc_method)
        end = timer()

        acc.append(metrics.evaluate(arrLabel, arrPred))
        t.append(end-ini)
        mem.append(memory)
        f1.append(metrics.F1(arrLabel, arrPred))
        cf[0].append(arrFP[str(classToMonitor)])
        cf[1].append(arrFN[str(classToMonitor)])
        cf[2].append(arrTP[str(classToMonitor)])
        cf[3].append(arrTN[str(classToMonitor)])

    avg_acc.append(np.mean(acc))
    avg_time.append(np.mean(t))
    avg_memory.append(np.mean(mem))
    avg_F1.append(np.mean(f1))
    avg_cf.append([np.mean(cf[0]), np.mean(cf[1]), np.mean(cf[2]), np.mean(cf[3])])
    datasets.append('GTSRB')

    return avg_acc, avg_time, avg_cf, avg_memory, avg_F1, datasets