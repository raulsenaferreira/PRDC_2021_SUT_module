self = GAN_GTRSB.ALOCC_Model(
	dataset_name='mnist', trainPath=trainPath, attention_label=classToMonitor, 
	input_height=input_height,input_width=input_width, output_height=output_height,
	output_width=output_width, c_dim = channels
	)
self.train(epochs=10, batch_size=32, sample_interval=500)

#self.adversarial_model.load_weights('./checkpoint/ALOCC_Model_6.h5')

testPath='data'+sep+'GTS_dataset'+sep+"kaggle"+sep
#X_test, y_test = util.load_GTRSB_csv(testPath, "Test.csv")
X_train,X_valid,y_train,Y_valid = util.load_GTRSB_dataset(input_height, input_width, channels, trainPath, 0.3, False)
X_test = X_train
y_test = y_train
def test_reconstruction(label, data_index = 11):
    specific_idx = np.where(y_test == label)[0]
    #print(specific_idx)
    if data_index >= len(X_test):
        data_index = 0
    data = X_test[specific_idx]#.reshape(-1, 28, 28, 3)[data_index:data_index+1]
    data = data[1]
    model_predicts = self.adversarial_model.predict(np.asarray([data]))
    
    fig= plt.figure(figsize=(5, 5))
    columns = 1
    rows = 2
    fig.add_subplot(rows, columns, 1)
    input_image = data.reshape(input_shape)
    reconstructed_image = model_predicts[0].reshape(input_shape)
    plt.title('Input')
    plt.imshow(input_image, label='Input')
    fig.add_subplot(rows, columns, 2)
    plt.title('Reconstruction')
    plt.imshow(reconstructed_image, label='Reconstructed')
    plt.show()
    # Compute the mean binary_crossentropy loss of reconstructed image.
    y_true = K.variable(reconstructed_image)
    y_pred = K.variable(input_image)
    error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
    print('Reconstruction loss:', error)
    print('Discriminator Output:', model_predicts[1][0][0])


test_reconstruction(classToMonitor)