'''

NLU net with Tensorflow

Version 0.0

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer



if __name__=="__main__":

    # Input and parameters
    N = 100
    phrases_to_process = [ 'The pen is on the table', 'The pen is under the table', 'Over the table there is a pen']

    # Create an instance of the tokenizer
    tokenizer = Tokenizer(num_words=N)
    tokenizer.fit_on_texts(phrases_to_process)
    indices = tokenizer.word_index
    print(indices)
