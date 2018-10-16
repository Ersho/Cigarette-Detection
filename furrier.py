import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cigarette.jpg',0)

"""
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
"""

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])) 

rows, cols = img.shape
mask = np.ones((rows,cols,2), np.uint8)
mask[rows//2 - 30: rows//2 + 30, cols//2 - 30 : cols //2 +30] = 0

fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude spectrum'), plt.xticks([]),plt.yticks([])
plt.subplot(123), plt.imshow(img_back, cmap='gray')
plt.title('Back image'), plt.xticks([]),plt.yticks([])


