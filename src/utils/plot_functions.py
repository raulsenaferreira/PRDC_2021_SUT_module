import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as ani
import numpy as np
import matplotlib.patches as mpatches
import pickle
from src.utils import util
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



def plotDistributions(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)

    for X in distributions:
        #reducing to 2-dimensional data
        x=classifiers.pca(X, 2)

        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plotDistributionByClass(instances, indexesByClass):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)

    for c, indexes in indexesByClass.items():
        X = instances[indexes]
        #reducing to 2-dimensional data
        x=classifiers.pca(X, 2)

        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plotAccuracy(arr, steps, label):
    arr = np.array(arr)
    c = range(len(arr))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arr, 'k')
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(0, steps+1, 10))
    plt.title(label)
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotDistributionss(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['Class 1', 'Class 2']
    ax = fig.add_subplot(121)

    for k, v in distributions.items():
        points = distributions[k]

        handles.append(ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plot(X, y, coreX, coreY, t):
    classes = list(set(y))
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        corePoints = coreX[np.where(coreY==cl)[0]]
        coreX1 = corePoints[:,0]
        coreX2 = corePoints[:,1]
        handles.append(ax.scatter(coreX1, coreX2, c = colors[color]))
        #labels
        classLabels.append('Class {}'.format(cl))
        classLabels.append('Core {}'.format(cl))
        color+=1

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def plot2(X, y, t, classes):
    X = classifiers.pca(X, 2)
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        #labels
        classLabels.append('Class {}'.format(cl))

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def finalEvaluation(arrAcc, steps, label):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, steps, label)


def plotF1(arrF1, steps, label):
    arrF1 = np.array(arrF1)
    c = range(len(arrF1))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arrF1, 'k')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if steps > 10:
        plt.xticks(range(1, steps+1, 10))
    else:
        plt.xticks(range(1, steps+1))
    plt.title(label)
    plt.ylabel("F1")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBoxplot(mode, data, labels):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=90)

    if mode == 'acc':
        plt.title("Accuracy - Boxplot")
        #plt.xlabel('step (s)')
        plt.ylabel('Accuracy')
    elif mode == 'mcc':
        plt.title('Mathews Correlation Coefficient - Boxplot')
        plt.ylabel("Mathews Correlation Coefficient")
    elif mode == 'f1':
        plt.title('F1 - Boxplot')
        plt.ylabel("F1")

    plt.show()


def plotAccuracyCurves(listOfAccuracies, listOfMethods):
    limit = len(listOfAccuracies[0])+1

    for acc in listOfAccuracies:
        acc = np.array(acc)
        c = range(len(acc))
        ax = plt.axes()
        ax.plot(c, acc)

    plt.title("Accuracy curve")
    plt.legend(listOfMethods)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.xticks(range(0, limit, 10))
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBars(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l], label=listOfMethods[l], align='center')

    plt.title("Execution time to perform all stream")
    plt.legend(listOfMethods)
    plt.xlabel("Methods")
    plt.ylabel("Execution time")
    plt.xticks(range(len(listOfTimes)))
    plt.show()


def plotBars2(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l])

    plt.title("Average Accuracy")
    plt.xlabel("Methods")
    plt.ylabel("Accuracy")
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfTimes)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def plotBars3(listOfAccuracies, listOfMethods):
    
    for l in range(len(listOfAccuracies)):    
        ax = plt.axes()
        ax.bar(l, 100-listOfAccuracies[l])

    plt.title("Average Error")
    plt.xlabel("Methods")
    plt.ylabel("Error")
    #plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfAccuracies)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def plotBars4(baseline, listOfAccuracies, listOfMethods):
    
    for l in range(1,len(listOfAccuracies)):    
        ax = plt.axes()
        #ax.bar(l, (listOfAccuracies[l]-baseline)/listOfAccuracies[l])
        ax.bar(l, ((listOfAccuracies[l]-baseline)/baseline)*100)
        print('Error reduction:',((listOfAccuracies[l]-baseline)/baseline)*100)

    plt.title("Reduction Percentage Error")
    plt.xlabel("Methods")
    plt.ylabel("% Error under baseline (Static SSL)")
    #plt.yticks(range(0, 101, 10))
    plt.xticks(range(1, len(listOfAccuracies)), listOfMethods[1:])
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


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


def run_act_func_animation(model, dataset_name, instances, labels, first_nth_classes, layerIndex, steps, file):
    
    fig = plt.figure()

    def plot_animation(i):
        plt.clf()
        uniform_data = []
        
        for c in range(first_nth_classes):
            ind_class = np.where(labels == c)
            image = np.asarray([instances[ind_class][i]])
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            #print(np.shape(arrWeights))
            uniform_data.append(arrWeights[:first_nth_classes])
        
        ax = sns.heatmap(uniform_data)#, annot=True
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Classes')
        title = 'Activation function of instance {} on GTSRB'.format(i, dataset_name)
        plt.title(title)
        
    animator = ani.FuncAnimation(fig, plot_animation, frames=200, interval = steps)
    animator.save(file, fps=2)
    #plt.show()


### Helper function for run_boxes_analysis()
def print_points_boxes(ax, c, boxes, arr_points, arr_pred, tau=0.0001, dim_reduc_obj=None):
    color={0:'yellow', 1:'green', 2:'blue'}
    arr_polygons = []

    for box in boxes:
        #print(class_to_monitor, box)
        x1 = box[0][0]
        x2 = box[0][1]
        y1 = box[1][0]
        y2 = box[1][1]

        x1 = x1*tau-x1 if x1 > 0 else x1-tau
        x2 = x2*tau+x2 if x2 > 0 else x2+tau
        y1 = y1*tau-y1 if y1 > 0 else y1-tau
        y2 = y2*tau+y2 if y2 > 0 else y2+tau

        rectangle = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        #print(rectangle)
        polygon = Polygon(rectangle)
        arr_polygons.append(polygon)

        ax.add_patch(mpatches.Polygon(rectangle, alpha=0.2, color=color[c]))

    for ypred, intermediateValues in zip(arr_pred, arr_points):
        x,y = None, None
        data = np.asarray(intermediateValues)
        #print(np.shape(data))
        
        if dim_reduc_obj!=None:
            dim_reduc_obj = pickle.load(open(dim_reduc_obj, "rb"))
            data = dim_reduc_obj.transform(data.reshape(1, -1))[0] #last version
            #print(np.shape(data))
            x = data[0]
            y = data[1]
        else:
            x = data[0]
            y = data[-1]

        point = Point(x, y)
        is_outside_of_box = True

        for polygon in arr_polygons:
            if polygon.contains(point):
                is_outside_of_box = False
        
        if is_outside_of_box:
            if c != ypred:
                #true positive
                plt.plot([x], [y], marker='.', markersize=10, color="red")
            else:
                #false positive
                plt.plot([x], [y], marker='x', markersize=10, color="red")
        else:
            if c == ypred:
                #true negative
                plt.plot([x], [y], marker='.', markersize=10, color=color[c]) 
            else:
                #false negative
                plt.plot([x], [y], marker='x', markersize=10, color=color[c])


### Function that performs analysis of positives and negatives detections from the OOB-based safety monitors
def run_boxes_analysis(model, dataset_name, technique, instances, labels,\
 first_nth_classes, layerIndex, steps, file, monitor_folder, dim_reduc_obj):
    num_instances = 50
    tau = 0.01 # enlarging factor for abstraction boxes area

    fig, ax = plt.subplots()

    for c in range(first_nth_classes):
        ind_class = np.where(labels == c)
        arr_points = []
        arr_pred = []
        #dim_reduc_obj = os.path.join(monitor_folder, dim_reduc_obj)

        for i in range(num_instances):
            image = np.asarray([instances[ind_class][i]])
            y_pred = np.argmax(model.predict(image))
            arr_pred.append(y_pred)

            boxes_path = os.path.join(monitor_folder, 'class_{}'.format(c), 'monitor_{}_3_clusters.p'.format(technique))
            boxes = pickle.load(open(boxes_path, "rb"))
            #print('boxes shape for class', c, np.shape(boxes))
            
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            arr_points.append(arrWeights)

        print_points_boxes(ax, c, boxes, arr_points, arr_pred, tau, dim_reduc_obj)

    plt.show()


### Helper function for plot_single_clf_pca_actFunc_based_analysis()
def startAnimation(X, y, yt, clf):
    X = np.array(X)
    y = np.array(y)
    yt = np.array(yt)
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
        
    # initialization function: plot the background of each frame
    def init():
        scatter = plt.scatter([], [], s=20, edgecolor='k')
        return scatter,
        
    # animation function.  This is called sequentially
    def animate(i):
        #print('X', np.shape(X))
        #print('y', np.shape(y))
        #print('yt', np.shape(yt))
        
        #decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=yt, s=30)
        cores = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, marker ='v', edgecolor='k')
        plt.title("Class {}".format(i+1))
        #plt.show()
        return scatter,
    
    anim = ani.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=100, blit=True)
    anim.save('animation.mp4', fps=1)
    #plt.show()


### Function that analyzes single classifiers trained on 2D projections from activation funcs
def plot_single_clf_pca_actFunc_based_analysis(model, dataset_name, clf, instances, labels,\
 first_nth_classes, layerIndex, steps, file, dim_reduc_obj):
    
    #fig, ax = plt.subplots()
    #fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8)) 
    '''
    for c in range(first_nth_classes):
        ind_class = np.where(labels == c)
        arr_pred_CNN = []
        arr_pred_monitor = []
        data = []

        for i in range(num_instances):
            image = np.asarray([instances[ind_class][i]])
            y_pred = np.argmax(model.predict(image))
            arr_pred_CNN.append(y_pred)
            
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            arrWeights = np.array(arrWeights).reshape(1, -1)
            
            reduced_data = dim_reduc_obj.transform(arrWeights)
            data.append(reduced_data[0])
            m_pred = clf.predict(reduced_data)
            arr_pred_monitor.append(m_pred[0])
    '''
        #startAnimation(data, arr_pred_CNN, arr_pred_monitor, clf)
    #X = np.array(data)
    #y = np.array(arr_pred_CNN)
    #yt = np.array(arr_pred_monitor)
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
        
    # initialization function: plot the background of each frame
    def init():
        scatter = plt.scatter([], [], s=20, edgecolor='k')
        return scatter,

    def animate(i):
        num_instances = 10
        ind_class = np.where(labels == i)
        arr_pred_CNN = []
        arr_pred_monitor = []
        data = []

        for n in range(num_instances):
            image = np.asarray([instances[ind_class][n]])
            y_pred = np.argmax(model.predict(image))
            arr_pred_CNN.append(y_pred)
            
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            arrWeights = np.array(arrWeights).reshape(1, -1)
            
            reduced_data = dim_reduc_obj.transform(arrWeights)
            data.append(reduced_data[0])
            m_pred = clf.predict(reduced_data)
            arr_pred_monitor.append(m_pred[0])
        
        X = np.array(data)
        y = np.array(arr_pred_CNN)
        yt = np.array(arr_pred_monitor)
        #decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=yt, s=30)
        cores = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, marker ='v', edgecolor='k')
        plt.title("Class {}".format(i))
        plt.show()
        return scatter,
    for i in range(3):
        animate(i)
    #anim = ani.FuncAnimation(fig, animate, init_func=init, frames=30, interval=first_nth_classes, blit=True)
    #anim.save('animation.mp4', fps=1)

    #plt.show()