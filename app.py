import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, abort
from PIL import Image
import pickle

application = Flask(__name__)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

meta_file = r'datasets\cifar-100-python\meta'
meta_info = unpickle(meta_file)

labels100 = meta_info['fine_label_names']
labels10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(labels100)


# print(labels)

#Load cifar-100 model
cifar100_model = tf.keras.models.load_model('./model/cifar100')
cifar10_model = tf.keras.models.load_model('./model/cifar10')

with open('model/history/cifar10_history', "rb") as file_pi:
    cifar10_history = pickle.load(file_pi)

with open('model/history/cifar100_history', "rb") as file_pi:
    cifar100_history = pickle.load(file_pi)

def reshape_img(key):
    img = Image.open(f'img/{key}')
    img = (np.array(img))
    img = img / 255.0
    img = (np.expand_dims(img, 0))
    return img

# /predict endpoint
@application.route('/cifar100', methods=['POST'])
def cifar100():
    key = request.get_json()['key']
    if key is None:
        abort(400)

    img = reshape_img(key)
    prediction = cifar100_model.predict(img)
    prediction_per = prediction.flatten()
    prediction = np.array(prediction.argsort()).flatten()
    print('\nCIFAR-100:\n')
    print('Predicción 1: ', labels100[prediction[-1]], '(' + str(round(prediction_per[prediction[-1]]*100, 4)) + ' %)')
    print('Predicción 2: ', labels100[prediction[-2]], '(' + str(round(prediction_per[prediction[-2]]*100, 4)) + ' %)')
    print('Predicción 3: ', labels100[prediction[-3]], '(' + str(round(prediction_per[prediction[-3]]*100, 4)) + ' %)\n')
    return Response(status=200)

@application.route('/cifar10', methods=['POST'])
def cifar10():
    key = request.get_json()['key']
    if key is None:
        abort(400)

    img = reshape_img(key)
    prediction = cifar10_model.predict(img)
    prediction_per = prediction.flatten()
    prediction = np.array(prediction.argsort()).flatten()
    print('\nCIFAR-10:\n')
    print('Predicción 1: ', labels10[prediction[-1]], '(' + str(round(prediction_per[prediction[-1]]*100, 4)) + ' %)')
    print('Predicción 2: ', labels10[prediction[-2]], '(' + str(round(prediction_per[prediction[-2]]*100, 4)) + ' %)')
    print('Predicción 3: ', labels10[prediction[-3]], '(' + str(round(prediction_per[prediction[-3]]*100, 4)) + ' %)\n')
    return Response(status=200)

# Run the app
if __name__ == '__main__':
    application.debug = True
    application.run()

