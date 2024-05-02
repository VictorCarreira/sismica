import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt


### LEITURA DO ARQUIVO SEGY ###
segyfile = '../inputs/0219-0258-00-migpstm-20061201.sgy'
f = segyio.open(segyfile, ignore_geometry=True)


# CRIAÇÃO DE UMA MATRIX SISMICA E PLOTAGEM
seismic = f.trace.raw[:].T
scale = 0.5*np.std(seismic)
plt.figure(1, figsize=(15,5))
plt.imshow(seismic, aspect="auto", cmap='Greys', vmin=-scale, vmax=scale)
plt.savefig('seismic.png')


# SELECIONA O TRAÇO E PLOTA
print(f'Dimensão da matriz sísmica: {seismic.shape}')
print(f'Traço sísmico escolhido (1200):{seismic[:,4620]}')
traco=seismic[:,4620]
plt.figure(2, figsize=(15,5))
plt.plot(traco,'k')
plt.savefig('traco.png')

# PASSA-BANDA DO TRAÇO E PLOTA:


# Frequências de corte para o filtro passa-banda (em Hz)
lowcut = 10.0
highcut = 50.0

# Frequência de amostragem do sinal (em Hz)
fs = 1000.0

# Ordem do filtro
order = 6

# Cria os coeficientes do filtro passa-banda
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(order, [low, high], btype='bandpass')

# Aplica o filtro ao sinal
traco_filtrado = filtfilt(b, a, traco)

# Plota o sinal filtrado
plt.figure(figsize=(15, 5))
plt.plot(traco_filtrado, 'k')
plt.title('Traço Filtrado')
plt.savefig('traco_filtrado.png')


# Cria uma figura com dois subplots lado a lado
fig, axs = plt.subplots(1, 2, figsize=(15,5))

# Plota o traço no primeiro subplot
axs[0].plot(traco,'k')
axs[0].set_title('Traço')


axs[1].plot(traco_filtrado,'k')
axs[1].set_title('Traço filtrado')

# Salva a figura
plt.savefig('traco_e_traco_filtrado.png')


#FFT DO TRAÇO E PLOTA


fft_traco_filtrado = np.fft.fft(traco_filtrado)# Calcula a FFT do sinal filtrado


espectro_potencia = np.abs(fft_traco_filtrado)**2# Calcula o espectro de potência radial

# Plota o espectro de potência radial
plt.figure(figsize=(15, 5))
plt.plot(espectro_potencia, 'k')
plt.title('Espectro de Potência Radial')
plt.savefig('espectro_potencia.png')

# Cria uma figura com dois subplots lado a lado
fig, axs = plt.subplots(1, 2, figsize=(15,5))

# Plota o traço no primeiro subplot
axs[0].plot(espectro_potencia,'k')
axs[0].set_title('Espectro de Potência Radial')

# Plota o FFT do traço no segundo subplot
# Note que estamos plotando o módulo do FFT, pois o FFT é complexo
axs[1].plot(traco_filtrado,'k')
axs[1].set_title('Traço filtrado')

# Salva a figura
plt.savefig('traco_filtrado_e_espectro.png')

