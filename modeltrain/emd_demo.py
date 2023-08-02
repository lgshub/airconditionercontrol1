from PyEMD import EMD
import pylab as plt
import numpy  as np
s = np.random.random(100)
emd = EMD()
IMF = emd.emd(s)
N = IMF.shape[0]+1

# Plot results
plt.subplot(N,1,1)
plt.plot(s, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.savefig('simple_example')
plt.show()
