'''

NLU net with Tensorflow

Version 0.0

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.ops.gen_array_ops import shape


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

if __name__=="__main__":

    # Input and parameters
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_perc = 90
    num_epochs = 100

    ### DATA LOADING ###
    df_ture = pd.read_csv('./archive/True.csv')
    df_fake = pd.read_csv('./archive/Fake.csv')

    #TODO normalize text value and check possible missing values

    # Add lables
    df_ture['Fake'] = [0]*df_ture.shape[0]
    df_fake['Fake'] = [1]*df_fake.shape[0]

    # Join the two datasets
    dataset = pd.concat([df_ture, df_fake], axis=0)
    
    # Shuffle the DataFrame rows
    dataset = dataset.sample(frac = 1)

    # Create training and testing dataset
    val = round(dataset.shape[0]*training_perc/100)
    features = ['title']
    target = ['Fake']
    
    train_dataset = dataset[:val]
    test_dataset = dataset[val:]

    # Tokenization
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_dataset['title'])

    word_index = tokenizer.word_index

    # Labels
    training_labels = train_dataset['Fake']
    testing_labels = test_dataset['Fake'].to_numpy()

    # Features
    training_sequences = tokenizer.texts_to_sequences(train_dataset['title'])
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(test_dataset['title'])
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    ### TRAINING PHASE ###

    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)
    
    # Plot training results
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    