# load audio
# fft ---> spectrum
# stft ---> spectogram
# mfcc

#TODO: create functions
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
hop_length = 512
n_fft = 2048
sr = 22050

FIG_SIZE = (15,10)

#Write Filename in a music file which is in project folder
file = "please enter your song directory"

signal, sr = librosa.load(file)

#Audio WavePlot
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()

# perform Fourier transform
fft = np.fft.fft(signal)

# calculate abs values on complex numbers to get magnitude
spectrum = np.abs(fft)
# create frequency variable
f = np.linspace(0, sr, len(spectrum))


plt.plot(f,spectrum)   # this gives a symetrical result
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Magnitude")
plt.show()


# take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]


# plot spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()


# STFT -> spectrogram
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)


plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.show()


# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(signal, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

plt.show()



















