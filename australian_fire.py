import numpy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Licensed and Made BY ARDA DUMANOGLU 
## All rights reserved.

# making numpy values easy to read

np.set_printoptions(precision=3, suppress=True)

# Data loading with pandas

australian_fire_dataset = pd.read_csv("australian_fire.csv")

# separating features and labels for training  x = feature = independent variable    y =  label = dependent variable
# WE EXTRACTING DEPENDENT(LABEL) AND INDEPENDENT(FEATURE) VARIABLES.

x = australian_fire_dataset.copy()
x = np.array(x)   # load it on the numpy array
x = np.delete(x, numpy.s_[5:7], 1) # we delete acq_date and time which is unnecessary features for our training models. Their numeric values means nothing for us.
x = np.delete(x, numpy.s_[6], 1) # we delete instrument column Which is only 1 unique value MODIS since all of them are the same
x = np.delete(x, numpy.s_[6:8], 1) # we delete version which is all of the same and doesnt affect "6.0NRT" and also we delete the CONFİDENCE LABEL in our features.

# ENCODING CATEGORICAL DATAS  ( DAYNIGHT AND SATELLITE ) 5. ve 8 th indexes.
label_encoder= LabelEncoder()
x[:, 5]= label_encoder.fit_transform(x[:, 5])  # we encoding the Satellite values to TERRA  = 1  and AQUA = 0
x[:, 8]= label_encoder.fit_transform(x[:, 8])  # we encoding the Daynight values to Day = 0 and Night = 1


# To extract an independent variable, we will use iloc[ ] method of Pandas library.
# It is used to extract the required rows and columns from the dataset.

y = australian_fire_dataset.iloc[:, 9].values   # we only take the confidence column

# We convert it because i got the error which is "failed to convert a numpy array to tensor object" so i convert the float32 type and array
x = np.asarray(x).astype('float32')

# It is just basic preprocessing for normalization inputs for CSV data files.
normalize = preprocessing.Normalization()
normalize.adapt(x)

# SPLITTING TRAINING AND TEST DATASET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

# Since we need an one input for each column I choose the keras.sequential model

Australian_fire_Model = tf.keras.Sequential([   # Sequential model gets one input and one  output (and sequantial model is also called the "list of layers".
    normalize,
    layers.Dense(256,activation="relu",name="layer1"),     # LAYERS  converts input to  output . Model  looks like a big layer.
    layers.Dense(192,activation="relu",name="layer2"),
    layers.Dense(128,activation="relu",name="layer3"),
    layers.Dense(100,activation="relu", name="layer4")  # OUTPUT LAYER WE HAVE 100 OUTPUTS  Ratings from 0-100.
    # RELU IS DEFAULT TO GO. BUT WE CAN GO TO PROBABILITY DISTRIBUTION WITH SOFTMAX BUT THAT IS NOT OUR CASE
])
Australian_fire_Model.compile(loss=tf.losses.MeanSquaredError(),
                              optimizer=tf.optimizers.Adam())

# for train a model
Australian_fire_Model.fit(x_train, y_train, epochs=100)    # BATCH SIZE = NUMBER OF THE ROWS (HOW MANY ROWS WILL BE RETURNED AT ONCE)
