import numpy as np
import matplotlib.pyplot as plt

npix = 4

assert(npix % 2 != 0)
mean = np.zeros(npix * npix)
cov = np.identity(npix * npix)
result1 = np.random.multivariate_normal(mean, cov, 2)
a = result1[0]
b = result1[1]
result1 = np.array([complex(a[i], b[i]) for i in range(npix * npix)])

#result2 = np.array(result1)
#for j in range((npix * npix)/2):
#	result2[j] =result2[-1 - j].conjugate()
#result2 = result2.reshape ((npix, npix))

result2 = result1.reshape((npix, npix))
#origin = (npix / 2, npix / 2)
origin = (0,0)
result2[origin] = result2[origin].real

result2[1::-1,-1:npix/2-1:-1] = result2[1:,1:npix/2].conj()
result2[:npix/2-1:-1,0] = result2[1:npix/2,0].conj()
result2[0,:npix/2-1:-1] = result2[0,1:npix/2].conj()
if npix % 2 == 0:
    result2[npix/2] = result2[npix/2].real
    result2[:,npix/2] = result2[:,npix/2].real
    #result2[1:,1:npix/2] = result2[-2::-1,-1:npix/2-1:-1].conj()

import IPython; IPython.embed()

#k_radius = 3
#for i in np.arange(npix):
#	for j in np.arange(npix):
#		displacement = np.sqrt(((i - origin) ** 2) + ((j - origin) ** 2))
#		k_ring

result3 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result2)))
print 'hey'
print result3

plt.figure(1)
plt.imshow(result2.imag, interpolation='nearest')
plt.title('Random Fourier Field')

plt.figure(2)
plt.imshow(result3.real, interpolation='nearest')
plt.title('Configuration Space')
plt.show()
