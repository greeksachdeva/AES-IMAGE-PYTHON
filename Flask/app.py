import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import time
import sys
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256

keyInput = "MySecretKey123"
salt = b'MyFixedSalt123'
keySize = 16
ivSize = AES.block_size


app = Flask(__name__)

model = load_model('brain_tumor_classification_VGG16.h5')
# model1 = load_model('brain21.h5')

def get_className(classNo1):
    if classNo1 == 1:
        return "Brain tumor"
    elif classNo1 == 2:
        return "Not a brain tumor"
    elif classNo1 == 3:
        return "Not a brain image"
    
    
def encrypt(file_path):
        

        start_time = time.time()
        
        imageOrig = cv2.imread(file_path)
        rowOrig, columnOrig, depthOrig = imageOrig.shape

        minWidth = (AES.block_size + AES.block_size) // depthOrig + 1
        if columnOrig < minWidth:
            print('The minimum width of the image must be {} pixels, so that IV and padding can be stored in a single additional row!'.format(minWidth))
            sys.exit()

        imageOrigBytes = imageOrig.tobytes()
        
        key = PBKDF2(keyInput, salt, dkLen=16)

        hash_obj = SHA256.new(key)
        iv = hash_obj.digest()[:ivSize]

        cipher = AES.new(key, AES.MODE_CBC, iv)
        imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
        ciphertext = cipher.encrypt(imageOrigBytesPadded)

        paddedSize = len(imageOrigBytesPadded) - len(imageOrigBytes)
        void = columnOrig * depthOrig - ivSize - paddedSize
        ivCiphertextVoid = iv + ciphertext + bytes(void)
        imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype=imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)

        cv2.imwrite(file_path, imageEncrypted)
        cv2.destroyAllWindows()

def decrypt(file_path):

    key = PBKDF2(keyInput, salt, dkLen=16)
    
    hash_obj = SHA256.new(key)
    iv = hash_obj.digest()[:ivSize]
    
    imageEncrypted = cv2.imread(file_path)
    
    rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape
    rowOrig = rowEncrypted - 1
    encryptedBytes = imageEncrypted.tobytes()
    iv = encryptedBytes[:ivSize]
    imageOrigBytesSize = rowOrig * columnOrig * depthOrig
    paddedSize = (imageOrigBytesSize // AES.block_size + 1) * AES.block_size - imageOrigBytesSize
    encrypted = encryptedBytes[ivSize: ivSize + imageOrigBytesSize + paddedSize]
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decryptedImageBytesPadded = cipher.decrypt(encrypted)
    decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)
    
    decryptedImage = np.frombuffer(decryptedImageBytes, imageEncrypted.dtype).reshape(rowOrig, columnOrig, depthOrig)

    cv2.imwrite("decrypted_image.png", decryptedImage)
    cv2.destroyAllWindows()
    return decryptedImage


def getResult(img):
    # image = cv2.imread(img)
    image = decrypt(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    predict_x = model1.predict(input_img)
    result = np.argmax(predict_x, axis=1)
    print(result)
    if result == 0:
        predict_y = model.predict(input_img)
        output = np.argmax(predict_y, axis=1)
        print(output)
        if output == 1:
            return 1
        else:
            return 2
    else:
        return 3

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(_file_)
        
        filename="gfgfhf.png"
        file_path = os.path.join(basepath, 'uploads', filename)
        f.save(file_path)
        encrypt(file_path)
        
        value = getResult(file_path)
        
        res = get_className(value)
        return res

    else:
        value = getResult(file_path)
        output = get_className(value)
        return output

if __name__ == '_main_':
    app.run(port=3000, debug=False)