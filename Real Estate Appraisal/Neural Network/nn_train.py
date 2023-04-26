import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#https://www.geeksforgeeks.org/implementing-neural-networks-using-tensorflow/
#useful link

#https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
#You must ensure that the scale of your output variable matches the scale of the activation function (transfer function) on the output layer of your network.

df = pd.read_excel("data_cleaned/normalized_normal/cleanedAndTrimmedData.xlsx")

train_df = df.sample(frac=0.7, random_state=4)
other_df = df.drop(train_df.index)

val_df = other_df.sample(frac=0.5, random_state=4)
test_df = other_df.drop(val_df.index)

y_train = train_df.pop('SalePrice')
y_train_unscaled = train_df.pop('salePrice_unnormalized')
y_val = val_df.pop('SalePrice')
y_val_unscaled = val_df.pop('salePrice_unnormalized')
y_test = test_df.pop('SalePrice')
y_test_unscaled = test_df.pop('salePrice_unnormalized')

# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)

x_train = train_df
x_val = val_df
x_test = test_df

# print(x_train.shape)
# print(x_train.shape[1])

input_shape = [x_train.shape[1]]

model = tf.keras.Sequential([
 
    #best:51263.645262557075
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)

    #best:65000
    # tf.keras.layers.Dense(units=256, activation='relu', input_shape=input_shape),
    # tf.keras.layers.Dense(units=64, activation='relu'),
    # tf.keras.layers.Dense(units=64, activation='relu'),
    # tf.keras.layers.Dense(units=1)

    #best:65000
    # tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape),
    # tf.keras.layers.Dense(units=64, activation='relu'),
    # tf.keras.layers.Dense(units=128, activation='relu'),
    # tf.keras.layers.Dense(units=64, activation='relu'),
    # tf.keras.layers.Dense(units=64, activation='relu'),
    # tf.keras.layers.Dense(units=1)
])
model.summary()

# adam optimizer works pretty well for
# all kinds of problems and is a good starting point
model.compile(optimizer='adam', 
               
              # MAE error is good for
              # numerical predictions
              loss='mae'
              #metrics=[keras.metrics.SparseCategoricalAccuracy()],
              ) 

losses = model.fit(x_train, y_train,
 
                   validation_data=(x_val, y_val),
                    
                   # it will use 'batch_size' number
                   # of examples per example
                   batch_size=256,
                   epochs=30,  # total epoch
 
                   )

loss_df = pd.DataFrame(losses.history)
print(loss_df)

plt.plot(loss_df.loc[:,['loss','val_loss']])
plt.title("Loss of Neural Network During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Loss', 'Val_loss'])
plt.show()

mean = 90491.58821514217
std = 106496.3793332993

prediction = model.predict(x_test)

# correct2 = y_test.to_numpy()
# correct2 = correct2[:,np.newaxis]
# scaledCorrect2 = correct2*std + mean

correct = y_test_unscaled.to_numpy()
scaledCorrect = correct[:,np.newaxis]


scaledPrediction = prediction*std + mean


# print("scaledCorrect") #
# print(scaledCorrect)
# print(type(scaledCorrect))
# print(scaledCorrect.size)
# Name: SalePrice, Length: 438, dtype: float64
#<class 'pandas.core.series.Series'>
#(438,)

# print("scaledPrediction")
# print(scaledPrediction)
# print(type(scaledPrediction))
# print(scaledPrediction.shape)
#<class 'numpy.ndarray'>
#(438, 1)

plt.clf()
plt.scatter(scaledCorrect, scaledPrediction)
plt.plot([0,500000], [0,500000])
plt.title("Neural Network Predictions")
plt.xlabel("Correct Value")
plt.ylabel("Predicted Value")
plt.legend(['Predictions', 'Correct'])
plt.show()


dif = np.abs(scaledCorrect - scaledPrediction)
avgError = np.mean(dif)

correct = dif < 50000
accuracy = np.count_nonzero(correct) / correct.shape[0]
print(accuracy)

df_out = pd.DataFrame()
df_out["correct"] = scaledCorrect[:,0]
df_out["prediction"] = scaledPrediction[:,0]
df_out["withinThreshold"] = correct[:,0]
df_out.to_excel("nn_output.xlsx")


# print("prediction")
# print(prediction.shape)
# print(type(prediction))
# print("scaledPrediction")
# print(scaledPrediction.shape)
# print(type(scaledPrediction))
# print("correct")
# print(scaledCorrect.shape)
# print(type(scaledCorrect))
# print("avgError")
# print("avgError")
# print(avgError.shape)
# print(type(avgError))