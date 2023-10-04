import sys
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
# This program encrypts a jpg With AES-256. The encrypted image contains more data than the original image (e.g. because of 
# padding, IV etc.). Therefore the encrypted image has one row more. Supported are CBC and ECB mode.


# Set sizes
keySize = 16
ivSize = AES.block_size


keyInput = "9461160295"
# Convert key input to bytes
salt = b'MyFixedSalt123'  # Generate a random salt
key = PBKDF2(keyInput, salt, dkLen=16)  # Derive a 256-bit (32 bytes) key using PBKDF2


hash_obj = SHA256.new(key)
iv = hash_obj.digest()[:ivSize]

imageEncrypted = cv2.imread("upload.png")

#
# Start Decryption ----------------------------------------------------------------------------------------------
#

# Convert encrypted image data to ciphertext bytes
rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape 
rowOrig = rowEncrypted - 1
encryptedBytes = imageEncrypted.tobytes()
iv = encryptedBytes[:ivSize]
imageOrigBytesSize = rowOrig * columnOrig * depthOrig
paddedSize = (imageOrigBytesSize // AES.block_size + 1) * AES.block_size - imageOrigBytesSize
encrypted = encryptedBytes[ivSize : ivSize + imageOrigBytesSize + paddedSize]

# Decrypt
cipher = AES.new(key, AES.MODE_CBC, iv)
decryptedImageBytesPadded = cipher.decrypt(encrypted)
decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)

# Convert bytes to decrypted image data
decryptedImage = np.frombuffer(decryptedImageBytes, imageEncrypted.dtype).reshape(rowOrig, columnOrig, depthOrig)

# Display decrypted image
cv2.imwrite("decrypted_image.png", decryptedImage)
cv2.imshow("Decrypted Image", decryptedImage)
cv2.waitKey()

# Close all windows
cv2.destroyAllWindows()
