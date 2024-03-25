import os
import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer,TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.configs import *

class DataUtil:
    def __init__(self):
        pass

    def read_data(file_path:str,input_col:str,output_col:str,file_encoding:str='ISO-8859-1'):
            for _,_,files in os.walk(file_path):
                for file in files:
                    if file.endswith(".csv"):
                        fullpath = os.path.join(file_path,file)
                        return pd.read_csv(fullpath, names = [output_col,input_col],encoding=file_encoding).dropna().drop_duplicates()

    def get_labels(data:pd.DataFrame,output_col:str):
        return to_categorical(LabelEncoder().fit_transform(data[output_col]), num_classes=3)

    def get_sentences(data:pd.DataFrame,input_col:str):
        return BertTokenizer.from_pretrained(LLM_MODEL)(data[input_col].tolist(), return_tensors='np', \
            padding=True, truncation=True, max_length=512)

    def train_test_split(input,output,test_size=0.2):
        return train_test_split(input, output, test_size=test_size, random_state=5,stratify=output)

class FinModelUtil:
    def __init__(self):
        pass

    def __get_bert_model(num_classes=3):
        return TFBertModel.from_pretrained(LLM_MODEL,num_labels = num_classes,\
                                                               output_attentions = False, output_hidden_states = False)

    def train_model(model,inputs, labels, lr=0.01,n_epochs=1):
        model.fit(inputs, labels)
        return model
    
    def get_classifier_to_train(model='rf'):
        if(model=='rf'):
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif(model=='svc'):
            return SVC()

    def get_embeddings(data):
        return FinModelUtil.__get_bert_model().predict(data)



