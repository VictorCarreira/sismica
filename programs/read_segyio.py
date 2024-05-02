import segyio
import numpy as np
import matplotlib.pyplot as plt

segyfile = '0219-0258-00-migpstm-20061201.sgy'

f = segyio.open(segyfile, ignore_geometry=True)

seismic = f.trace.raw[:].T

scale = 0.5*np.std(seismic)

plt.figure(1, figsize=(15,5))

plt.imshow(seismic, aspect="auto", cmap='Greys', vmin=-scale, vmax=scale)

plt.savefig('seismic.png')