from pickle import Unpickler
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

train_file = r'datasets\cifar-100-python\train'
train_images = unpickle(train_file)

print(train_images.keys())

for item in train_images:
    print(item, type(train_images[item]))

X_train_images = train_images['data']
# Reshape the whole image data
X_train_images = X_train_images.reshape(len(X_train_images),3,32,32)
# Transpose the whole data
X_train_images = X_train_images.transpose(0,2,3,1)

X_train_labels = np.array(train_images['fine_labels'])

meta_file = r'datasets\cifar-100-python\meta'
meta_info = unpickle(meta_file)
labels = meta_info['fine_label_names']

# TEST
test_file = r'datasets\cifar-100-python\test'

test_images = unpickle(test_file)

X_test_images = test_images['data']
# Reshape the whole image data
X_test_images = X_test_images.reshape(len(X_test_images),3,32,32)
# Transpose the whole data
X_test_images = X_test_images.transpose(0,2,3,1)

X_test_labels = np.array(test_images['fine_labels'])


fig=plt.figure(figsize=(10, 8))
for i in range(1, 17):
    fig.add_subplot(4, 4, i)
    plt.imshow(X_train_images[i-1])
    plt.xticks([])
    plt.yticks([])
    plt.title("Label: {}"
            .format(labels[train_images['fine_labels'][i-1]]))
plt.show()

# NORMALIZATION
X_train_images = X_train_images / 255.0
X_test_images = X_test_images / 255.0

# Convertimos la columna de clase a categórica para poder usar data augmentation
X_train_labels = keras.utils.to_categorical(X_train_labels, 100)
X_test_labels = keras.utils.to_categorical(X_test_labels, 100)

# VALIDATION SPLIT
X_train, X_validation, y_train, y_validation = train_test_split(X_train_images, X_train_labels, test_size=0.1, random_state=5432)


# RED NEURONAL (9 CAPAS)
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

    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3), kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                epsilon=0.005, 
                beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3), kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                epsilon=0.005, 
                beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.2, seed=76564),   

    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3), kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                epsilon=0.005, 
                beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3), kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.BatchNormalization(momentum=0.95, 
                epsilon=0.005, 
                beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.2, seed=76564),

    keras.layers.Flatten(),
    
    keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform', use_bias=True,
                       bias_regularizer=keras.regularizers.L2(1e-4),
                       kernel_regularizer=keras.regularizers.L2(1e-4), 
                       activity_regularizer=keras.regularizers.L2(1e-5)),
    keras.layers.Dropout(0.65, seed=2023),

    keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', use_bias=True,
                       bias_regularizer=keras.regularizers.L2(1e-4),
                       kernel_regularizer=keras.regularizers.L2(1e-4), 
                       activity_regularizer=keras.regularizers.L2(1e-5)),  
    keras.layers.BatchNormalization(momentum=0.95, 
                epsilon=0.005, 
                beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
    keras.layers.Dropout(0.65, seed=2023),
 

    keras.layers.Dense(100, activation='softmax')
])

# Configuration for creating new images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
)

train_datagen.fit(X_train)
model.summary()

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.001)
rms = RMSprop(learning_rate=0.003) # probar con 0.01, 0.0001, use_bias & l2 en conv
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

#early stopping to monitor the validation loss and avoid overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, factor= 0.1, min_lr= 1e-6, verbose=1)

history = model.fit(train_datagen.flow(X_train, y_train, batch_size=128), epochs=250, steps_per_epoch=len(X_train)/128,
                    validation_data=(X_validation, y_validation), shuffle=True, callbacks=[early_stop, rlrop])
# history = model.fit(X_train_images, X_train_labels, batch_size=128, epochs=100, steps_per_epoch=len(X_train_images)/128,validation_split=0.1, shuffle=True, callbacks=[early_stop, rlrop])

test_loss, test_acc, auc = model.evaluate(X_test_images, X_test_labels, verbose=0)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)
print('\nTest AUC:', auc, '\n')

# Serialization
model.save('model/cifar100')
# Save history of training
with open('model/history/cifar100_history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


predictions = model.predict(X_test_images)

# print(predictions[1])

# Output: ?
print('predicted value:', labels[np.argmax(predictions[1])])

# Output: forest
print('true value:', labels[X_test_labels[1].argmax()])






