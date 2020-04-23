from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from src.utils import util
from src.MNIST_experiments import SCGAN_MNIST_monitor
from keras.models import load_model


def get_reconstruction_loss(img, adversarial_model):
    img = img.reshape(-1, 28, 28, 1)#[0]
    model_predicts = adversarial_model.predict(img)
    input_image = img.reshape((28, 28))
    reconstructed_image = model_predicts[0].reshape((28, 28))

    # Compute the mean binary_crossentropy loss of reconstructed image
    y_reconstructed = K.variable(reconstructed_image)
    y_pred = K.variable(input_image)
    #print("model_predicts", model_predicts)
    reconstruction_loss = K.eval(binary_crossentropy(y_reconstructed, y_pred)).mean()
    discriminator_output = model_predicts[1][0][0]
    #print("result", reconstruction_loss, discriminator_output)

    return reconstruction_loss, discriminator_output


def run(classToMonitor, models_folder, monitors_folder, model_name, monitor_name, data_index=11):
    count = [0, 0]
    arrPred = []
    reconstruction_loss_threshold = 0.8 
    label = classToMonitor

    arrFalseNegative = {str(classToMonitor): 0}
    arrTrueNegative = {str(classToMonitor): 0}
    arrFalsePositive = {str(classToMonitor): 0}
    arrTruePositive = {str(classToMonitor): 0}
    
    #loading test set, model, and monitor
    X_train, y_train, X_test, y_test, _ = util.load_mnist(onehotencoder=False)
    model = load_model(models_folder+model_name)

    self = SCGAN_MNIST_monitor.ALOCC_Model(input_height=28,input_width=28)
    self.adversarial_model.load_weights(monitors_folder+monitor_name)

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))
    
    for img, lab in zip(X_test, y_test):
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        #predicting
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)

        #monitoring
        reconstruction_loss, discriminator_output = get_reconstruction_loss(img, self.adversarial_model)

        if yPred == classToMonitor:
            if reconstruction_loss <= reconstruction_loss_threshold:
                count[0] += 1
                if yPred != lab:
                    arrFalseNegative[str(classToMonitor)] += 1 #False negative  
                    #print("False negative !!\n\n")        
                if yPred == lab: 
                    arrTrueNegative[str(classToMonitor)] += 1 #True negatives
            else:
                count[1] += 1
                if yPred != lab: 
                    arrTruePositive[str(classToMonitor)] += 1 #True positives
                if yPred == lab: 
                    arrFalsePositive[str(classToMonitor)] += 1 #False positives
                    #print("False positive !!\n\n")
        #elif lab==classToMonitor and yPred != classToMonitor:
            #print("missclassification --- new pattern for class",yPred, str(lab))
    
    return arrPred, count, arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative