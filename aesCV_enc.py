import time
import sys
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
import base64
# Set sizes
keySize = 16  # 16 bytes for AES 128
ivSize = AES.block_size

# Record start time
start_time = time.time()

# Load original image
imageOrig = cv2.imread("yes3.jpg")
rowOrig, columnOrig, depthOrig = imageOrig.shape

# Check for minimum width
minWidth = (AES.block_size + AES.block_size) // depthOrig + 1
if columnOrig < minWidth:
    print('The minimum width of the image must be {} pixels, so that IV and padding can be stored in a single additional row!'.format(minWidth))
    sys.exit()

# Display original image
#cv2.imshow("Original image", imageOrig)
#cv2.waitKey()

# Convert original image data to bytes
imageOrigBytes = imageOrig.tobytes()

# Encrypt
# key = get_random_bytes(keySize)

keyInput = '9461160295'
# Convert key input to bytes
salt = b'MyFixedSalt123'  # Generate a random salt
key = PBKDF2(keyInput, salt, dkLen=16)  # Derive a 256-bit (32 bytes) key using PBKDF2

print(key);
hex_key = key.hex()  # Convert binary to hexadecimal
print(hex_key)
# ~ print(key);

hash_obj = SHA256.new(key)
iv = hash_obj.digest()[:ivSize]

cipher = AES.new(key, AES.MODE_CBC, iv)
imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
ciphertext = cipher.encrypt(imageOrigBytesPadded)

# Convert ciphertext bytes to encrypted image data
#    The additional row contains columnOrig * DepthOrig bytes. Of this, ivSize + paddedSize bytes are used
#    and void = columnOrig * DepthOrig - ivSize - paddedSize bytes unused
paddedSize = len(imageOrigBytesPadded) - len(imageOrigBytes)
void = columnOrig * depthOrig - ivSize - paddedSize
ivCiphertextVoid = iv + ciphertext + bytes(void)
imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype=imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)

# Display encrypted image
#cv2.imshow("Encrypted image", imageEncrypted)
#cv2.waitKey()

# Save the encrypted image (optional)
#    If the encrypted image is to be stored, a format must be chosen that does not change the data. Otherwise,
#    decryption is not possible after loading the encrypted image. bmp does not change the data, but jpg does.
#    When saving with imwrite, the format is controlled by the extension (.jpg, .bmp). The following works:
cv2.imwrite("encrypted_image.bmp", imageEncrypted)
#    imageEncrypted = cv2.imread("topsecretEnc.bmp")

# Close all windows
cv2.destroyAllWindows()

# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print the execution time
print("Execution time: {:.2f} seconds".format(execution_time))
