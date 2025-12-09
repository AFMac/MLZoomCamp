import onnxruntime as ort
import numpy as np
from keras_image_helper import create_preprocessor

onnx_model_path = 'hair_classifier_empty.onnx'
session = ort.InferenceSession(onnx_model_path, providers = ['CPUExecutionProvider'])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

def tensor(img_array):
    return (img_array/255)

def norm(img_arr):
    #ImageNet normalization values from prior assignment
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    img_arr[..., 0] -= means[0]
    img_arr[..., 0] /= stds[0]
    img_arr[..., 1] -= means[1]
    img_arr[..., 1] /= stds[1]
    img_arr[..., 2] -= means[2]
    img_arr[..., 2] /= stds[2]

    return img_arr

def preprocess(img_array):
    img_array = np.array(img_array).astype('float32')  #make usable by Onnx
    img_array = tensor(img_array)
    #print(f'Tensor shape: {img_array.shape}')
    img_array = norm(img_array)
    #print(f'Normalized Tensor shape: {img_array.shape}')
    return img_array

def lambda_handler(event, context):
    url = event['url']
    preprocessor = create_preprocessor(preprocess, target_size=(200, 200))
    X = preprocessor.from_url(url)
    X = np.transpose(X, (0, 3, 1, 2))

    session_run = session.run([output_name], {input_name: X})
    predictions = session_run[0][0].tolist()

    #result = dict(zip(classes, predictions))
    return predictions
