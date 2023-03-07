import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from Crypto.Cipher import AES
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

from django.shortcuts import render
from django.http import HttpResponse

BASE_DIR = Path(__file__).resolve().parent.parent
key = "the key is uchiha_itachi"

def encrypt_message(message, key):
    message = message.encode()
    padding = 16 - (len(message) % 16)
    message += bytes([padding] * padding)
    key = key[:16].encode()
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(message)
    return ciphertext

def decrypt_message(ciphertext, key):
    key = key[:16].encode()
    cipher = AES.new(key, AES.MODE_ECB)
    message = cipher.decrypt(ciphertext)
    padding = message[-1]
    message = message[:-padding]
    message = message.decode()
    return message


def index(request):
    return render(request,'index.html')

def predict(request):

    seats = request.POST['seats']
    engine = request.POST['engine']
    mileage = request.POST['mileage']
    km_driven = request.POST['km_driven']
    fuel_type = request.POST['fuel_type']
    owner_type = request.POST['owner_type']
    max_power = request.POST['max_power']
    seller_type = request.POST['seller_type']
    transmission_type = request.POST['transmission_type']
    
    arr = np.array([ seller_type, owner_type ,km_driven, fuel_type, transmission_type, mileage, engine, max_power, seats])
    arr = [ encrypt_message(x, key) for x in arr ]
    arr = [ int.from_bytes(x, byteorder='big') for x in arr ]
    arr = np.array(arr).reshape(1,-1)
    print(arr)

    filename = 'knn_model_for_encrypted_data.sav'
    file_path = os.path.join(BASE_DIR,f'static/model/{filename}')
    loaded_model = pickle.load(open(file_path, 'rb'))

    filename = 'price_transform.csv'
    file_path = os.path.join(BASE_DIR,f'static/csv/{filename}')
    df = pd.read_csv(file_path)
    df['CONVERTED']= label_encoder.fit_transform(df['selling_price_(in-lakhs)'])
    y_predict = loaded_model.predict(arr)
    pred = label_encoder.inverse_transform(y_predict)
    print('pred',pred)

    encrypted_message = int(pred[0]).to_bytes(16, byteorder='big')
    decrypted_message = decrypt_message(encrypted_message, key)
    predicted_car_prize =  float(decrypted_message)

    return render(request, 'predict.html',{'predicted_prize':predicted_car_prize})





