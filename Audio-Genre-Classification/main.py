import os

import tensorflow.keras as keras
import librosa
import math
import numpy as np
from cnn_genre_classifier import predict
def time_to_sample(time):
    return time*22050

def prepare_user_song(datapath, num_seg=20):
    """

    :param datapath: song datapath
    :param num_seg: segment no for each song
    :return: partition of song that will use to prediction
    """
    sr = 22050
    hop_length=512
    signal, sample_rate = librosa.load(datapath, sr=sr)
    signal = signal[time_to_sample(30):time_to_sample(60)] #dataset consists 30s songs. We cut song as 30s
    duration = librosa.get_duration(y=signal, sr=sr)

    SAMPLES_PER_TRACK = sr * duration
    samples_per_segment = int(SAMPLES_PER_TRACK / num_seg)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    mfcc_list = []
    for d in range(num_seg):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sr=sample_rate, n_mfcc=13, n_fft=2048,hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc_list.append(mfcc.tolist())
    mfcc_list = np.array(mfcc_list)

    partition = mfcc_list[10]
    partition_for_prediction = partition[..., np.newaxis]

    return partition_for_prediction

if __name__ == "__main__":
    song_path = input("Please enter your audio path: ")
    song = prepare_user_song(song_path)
    song_name = os.path.split(song_path)[-1]

    saved_model = keras.models.load_model("my_model")
    print(f"   --- Prediction on {song_name} ---")
    predict = predict(saved_model, song, "write your target genre here")