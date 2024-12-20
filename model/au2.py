#Audio analysis and processing for Machine learning

import numpy as np
import os
import json

from sklearn.model_selection import train_test_split
#import tensorflow.keras as keras
from tensorflow.keras.models import Sequential #necaserry?
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
#from tensorflow.keras.callbacks import TensorBoard #callback for epochs
import tensorflow.keras.callbacks
from keras.regularizers import l2

import pandas as pd
import matplotlib.pyplot as plt

#my_path = os.path.abspath(os.path.dirname(__file__))
JSON_PATH = "data.json"
output_filename = "test2_rev3"

#path to store model and files
output_file = f"./models/{output_filename}/{output_filename}"

#check if output directory exists. If not create directory.
if not os.path.exists(output_file):
    os.makedirs(output_file[:-len(output_filename)]) 
    print("Output directory created.")

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    
    #convert list to numpy array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y
  
def prepare_data(test_size, validation_size):
    
    #load data
    X, y = load_data(JSON_PATH)
    
    #create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    #create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    print("train data shape: ",X_train.shape)
    # 3D array - (bins, n_mfccs, depth =1)
    #add one more dimention
    X_train = X_train[..., np.newaxis] # 4D - (n_samples, bins(segments), n_mfccs, depth=1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print("new train data shape: ",X_train.shape)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test
    
def build_CNN_model(input_shape):
    #create sequential model = feed forward nerual network
    model = Sequential()
    
    #layer 1, 64 neurons, 
    model.add(Conv2D(32,(3,3),input_shape=input_shape, padding='same')) #first layer
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #layer 2 #pass input shape in each layer?!?!?
    model.add(Conv2D(64,(3,3), padding='same')) #second layer, no need for input shape second time
    model.add(Activation("relu"))
    model.add(MaxPooling2D((3,3),strides=(2,2),padding='same'))#pool_size=(2,2)))
    
    #layer 3 --added
    model.add(Conv2D(128,(3,3), padding='same')) #thrid layer, no need for input shape second time
    model.add(Activation("relu"))
    model.add(MaxPooling2D((3,3),strides=(2,2),padding='same'))#pool_size=(2,2)))
    
    #layer 4 --added
    model.add(Conv2D(256,(3,3), padding='same')) #thrid layer, no need for input shape second time
    model.add(Activation("relu"))
    model.add(MaxPooling2D((3,3),strides=(2,2),padding='same'))#pool_size=(2,2)))
    
    
    #dense layer --- flatten the 3d to 1d.
    model.add(Flatten())
    
    model.add(Dense(256, kernel_regularizer=l2(0.01)))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))  

    #model.add(Dense(64))
    #model.add(Activation("relu"))
    #model.add(Dropout(0.3))

    #output "layer" (activation, not really a layer ?)
    model.add(Dense(1, kernel_regularizer=l2(0.01)))
    model.add(Activation("sigmoid"))
    
    
    return model
    
def predict(model, X, y):
    X = X[np.newaxis, ...]
    
    prediction = model.predict(X)
    
    predicted_index = np.argmax(prediction, axis = 1) #wtf is this?!?
    print("Expected index: {}, Predicted index: {}".format(y, int(np.round(prediction))))
    print("Not rounded Prediction = ", prediction )
    #print("type of prediction ",type(prediction))
    #print("type of expectation ", type())
    if(y == int(np.round(prediction))):
        print("Correct")
    else:
        print("False prediction!")
    
if __name__ == "__main__":
    
    #create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(0.10, 0.10) #25, 20
    
    #build CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) #since depth = 1, isnt it unessacary to include it?!?
    model = build_CNN_model(input_shape)
    
    #compile network
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics = ['accuracy'])
    
    #train CNN
    history = model.fit(
    X_train, y_train,
    validation_data=(X_validation, y_validation),
    epochs= 40,
    batch_size=32, #semi 32
    callbacks=[
        tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
    #plot Loss/Accuracy, display model summary and save model
    pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
    plt.title("Accuracy")
    plt.savefig(f'{output_file}_accuracy.png')


    pd.DataFrame(history.history)[['loss','val_loss']].plot()
    plt.title("Loss")
    plt.savefig(f'{output_file}_loss.png')

    model.summary()
    #Using keras format
    model.save(f'{output_file}.keras')
    #Using keras HDF5, THIS IS LEGACY, use keras instead.
    #model.save('{output_file}.h5')


    #evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print("Accuracy on test set is: {}".format(test_accuracy))
    print("with error of {}".format(test_error))
    
    #Make prediction on a sample
    print("Testset lenght = ", len(X_test))
    i=0
    for i in range(len(X_test)):
        X = X_test[i]
        y = y_test[i]
        predict(model, X, y)
        i +=1
    
    plt.show()
    
    
    