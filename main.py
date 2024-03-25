from src.utils import *
from src.configs import*
# from src.model_custom import *
import pandas as pd
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
from sklearn.metrics import confusion_matrix, accuracy_score



if __name__ == "__main__":
    #Load data
    data_dir = os.path.join(os.getcwd(),'data') #The data should be present inside folder 'data' as csv
    data = DataUtil.read_data(data_dir,INPUT_COL,OUTPUT_COL,FILE_ENCODING)
    sentences = DataUtil.get_sentences(data,INPUT_COL)['input_ids'] #Tokenized sentences using Bert tokenizer
    labels = DataUtil.get_labels(data,OUTPUT_COL)
    x_train,x_test,y_train,y_test = DataUtil.train_test_split(sentences,labels)

    #Train model using bert embeddings
    model = FinModelUtil.get_classifier_to_train(model='rf') #Gets either random forest or svc model for training the data
    x_train = FinModelUtil.get_embeddings(x_train)[0][:, 0, :] #Gets the embeddings from Bert LLM
    model = FinModelUtil.train_model(model,x_train,y_train)

    # Make predictions on the test set
    x_test = FinModelUtil.get_embeddings(x_test)[0][:, 0, :] #Gets the embeddings from Bert LLM for test data
    predictions = model.predict(x_test) #Predict the test data
    predicted_labels = np.argmax(predictions, axis=1)

    # True labels are the argmax of the one-hot encoded labels
    true_labels = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:")
    print(accuracy)

    # print(sentences)
    # print(labels.shape)


