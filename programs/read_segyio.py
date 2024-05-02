import scipy.fft
import scipy.signal
import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.fft import fft

segyfile = '../inputs/0219-0258-00-migpstm-20061201.sgy'

f = segyio.open(segyfile, ignore_geometry=True)

seismic = f.trace.raw[:].T
print(seismic.shape)
print(round(seismic.shape[1]/2))

scale = 0.5*np.std(seismic)
plt.figure(1, figsize=(15,5))

plt.imshow(seismic, aspect="auto", cmap='Greys', vmin=-scale, vmax=scale)

plt.savefig('seismic.png')

traco = seismic[:, round(seismic.shape[1]/2)]
print(type(traco))
print(traco)
print('')
plt.figure(2, figsize=(15,5))
plt.plot(traco)
plt.savefig('traco.png')


traco_freq = np.abs(fft(traco))
print(traco_freq)
print('')
plt.figure(3, figsize=(15,5))
plt.plot(traco_freq)
plt.savefig('FFT_traco.png')

fs = 1000*0.5
# Design the bandpass filter
lowcut = 10  # Lower cutoff frequency
highcut = 50  # Upper cutoff frequency
order = 6  # Filter order
#b, a = butter(order, [lowcut/fs, highcut/fs], btype='band')
b, a = butter(order, [lowcut/fs, highcut/fs], btype='bandpass')
filtered = filtfilt(b, a, traco)
plt.figure(4, figsize=(15,5))
plt.plot(filtered)
plt.savefig('Filtered.png')

traco_freq_filter = np.abs(fft(filtered))
print(traco_freq_filter)
print('')
plt.figure(5, figsize=(15,5))
plt.plot(traco_freq_filter)
plt.savefig('FFT_traco_filtered.png')

pw_spec = np.abs(traco_freq)**2
plt.figure(6, figsize=(15,5))
plt.plot(pw_spec)
plt.savefig('power_spectrum.png')

pw_spec_filt = np.abs(traco_freq_filter)**2
plt.figure(7, figsize=(15,5))
plt.plot(pw_spec_filt)
plt.savefig('power_spectrum_filt.png')

# Generate some sample data
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
f1 = 50  # Frequency of first sinusoid
f2 = 120  # Frequency of second sinusoid
x = 0.7*np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)  # Signal with two sinusoids

# Perform FFT
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), 1/fs)
plt.figure(8, figsize=(15,5))
plt.plot(freqs, X)

# Compute power spectrum
power = np.abs(X)**2

# Plot power spectrum
plt.figure(9, figsize=(15,5))
plt.plot(freqs[:len(freqs)//2], power[:len(freqs)//2]) # LIMITADO
#plt.plot(freqs, power) # NÃO LIMITADO
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.grid(True)
plt.show()
plt.savefig('regular_power_spec.png')