import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    lastIndex = window_size
    i = 0
    count = len(series)
      
    while(lastIndex < count):
        X.append(list(series[i:lastIndex]))
        y.append([series[lastIndex]])
        
        i = i + 1
        lastIndex = lastIndex + 1

    X = np.asarray(X)
    y = np.asarray(y)
    y = np.reshape(y, (len(y),1)) #optional

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    model.add(Dense(1))
    print(model.summary())
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import string
    punctuation = ['!', ',', '.', ':', ';', '?']        
    Valid_Characters = string.ascii_letters
    uniq = ''.join(set(text))
    for c in uniq:
        if c not in punctuation and c not in Valid_Characters:
            text = text.replace(c,' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
       
    lastIndex = window_size
    i = 0
    count = len(text)

    while (lastIndex < count):
        inputs.append(text[i:lastIndex])
        outputs.append(text[lastIndex])
        i += step_size
        lastIndex += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
