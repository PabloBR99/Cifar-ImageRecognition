from pickle import Unpickler
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop


(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

# RESHAPE
x_train_full = x_train_full.reshape(x_train_full.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
# NORMALIZATION
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0
# Convertimos la columna de clase a categórica para poder usar data augmentation
y_train_full = keras.utils.to_categorical(y_train_full, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# VALIDATION SPLIT
x_train, x_validation, y_train, y_validation = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=5432)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


fig=plt.figure(figsize=(10, 8))
for i in range(1, 17):
    fig.add_subplot(4, 4, i)
    plt.imshow(x_train[i-1])
    plt.xticks([])
    plt.yticks([])
    plt.title("Label: {}"
            .format(labels[y_train[i-1].argmax()]))
plt.show()


# RED NEURONAL (8 CAPAS)
model = keras.Sequential([


    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3), kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                                    epsilon=0.005, 
                                    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                                    gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3), kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                                    epsilon=0.005, 
                                    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                                    gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.2, seed=76564),



    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                                    epsilon=0.005, 
                                    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                                    gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                                    epsilon=0.005, 
                                    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                                    gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.2, seed=76564),



    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                                    epsilon=0.005, 
                                    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                                    gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                                    epsilon=0.005, 
                                    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                                    gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.2, seed=76564),


    keras.layers.Flatten(),
    

    keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', use_bias=True,
                       bias_regularizer=keras.regularizers.L2(1e-4),
                       kernel_regularizer=keras.regularizers.L2(1e-4),
                       activity_regularizer=keras.regularizers.L2(1e-5)),
    keras.layers.Dropout(0.6, seed=2023),


    keras.layers.Dense(10, activation='softmax')
])

# Configuration for creating new images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    horizontal_flip=True,
)

train_datagen.fit(x_train)
model.summary()

# OPTIMIZADORES
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.002, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
rms = RMSprop(learning_rate=0.003) 
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

#early stopping to monitor the validation loss and avoid overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, factor= 0.5, min_lr= 1e-6, verbose=1)
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=128), epochs=100, steps_per_epoch=len(x_train)/128, validation_data=(x_test, y_test), callbacks=[early_stop, rlrop])
# history = model.fit(x_train, y_train, batch_size=128, epochs=20, steps_per_epoch=len(x_train)/128,validation_split=0.1, shuffle=True)

test_loss, test_acc, auc = model.evaluate(x_test, y_test, verbose=0)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)
print('\nTest AUC:', auc, '\n')

# Serialization
model.save('model/cifar10')
# Save history of training
with open('model/history/cifar10_history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

predictions = model.predict(x_test)

# print(predictions[1])

# Output: ?
print('predicted value:', labels[np.argmax(predictions[1])])

# Output: 
print('true value:', labels[y_test[1].argmax()])






