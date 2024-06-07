import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
import soundfile as sf
import os

rootdir = os.getcwd()
print(rootdir)

datadir = os.path.join(rootdir, "data")
print(datadir)

cleandir = os.path.join(datadir, "clean")
print(cleandir)

noisedir = os.path.join(datadir, "noise")
print(noisedir)

cleandata = os.listdir(cleandir)
print(cleandata)
noisedata = os.listdir(noisedir)
print(noisedata)

datalen = min(len(noisedata), len(cleandata))
print(datalen)


def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def add_noise(clean_audio, noise_audio, snr_db):
    noise_audio = noise_audio[:len(clean_audio)]
    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    noise_scaling_factor = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
    noisy_audio = clean_audio + noise_audio * noise_scaling_factor
    return noisy_audio

def split_audio(audio, sr=16000, duration=10):
    length = sr * duration
    chunks = [audio[i:i+length] for i in range(0, len(audio), length) if len(audio[i:i+length]) == length]
    return chunks

def extract_features(audio, sr=16000, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # (n_mfcc, time) -> (time, n_mfcc)로 변환

def build_autoencoder(input_shape):
    input_audio = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(input_audio)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, kernel_size=3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(8, kernel_size=3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.Model(input_audio, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

input_shape = (None, 13)  # MFCC 형상 (time, n_mfcc)
autoencoder = build_autoencoder(input_shape)
autoencoder.summary()

# clean_audio_files = [...]  # 깨끗한 음성 파일 리스트
# noise_audio_files = [...]  # 소음 파일 리스트
clean_audio_files = cleandata
noise_audio_files = noisedata

clean_audios = [load_audio(file) for file in clean_audio_files]
noise_audios = [load_audio(file) for file in noise_audio_files]

noisy_audios = [add_noise(clean, noise, snr_db=0) for clean, noise in zip(clean_audios, noise_audios)]
features = [extract_features(audio) for audio in noisy_audios]
clean_features = [extract_features(audio) for audio in clean_audios]

X_train = np.array(features)
y_train = np.array(clean_features)

autoencoder.fit(X_train, y_train, epochs=100, batch_size=16, shuffle=True)

# 모델 저장
model_path = "denoising_autoencoder.h5"
autoencoder.save(model_path)

# 테스트 데이터에 대한 소음 제거 및 저장
test_clean_audio_file = "path_to_test_clean_audio.wav"
test_noise_audio_file = "path_to_test_noise_audio.wav"

test_clean_audio = load_audio(test_clean_audio_file)
test_noise_audio = load_audio(test_noise_audio_file)
test_noisy_audio = add_noise(test_clean_audio, test_noise_audio, snr_db=0)
test_features = extract_features(test_noisy_audio)

# 모델 로드
autoencoder = tf.keras.models.load_model(model_path)

# 소음 제거 예측
denoised_features = autoencoder.predict(np.expand_dims(test_features, axis=0))
denoised_features = np.squeeze(denoised_features, axis=0)

# denoised_features를 오디오 신호로 변환 (역 MFCC 변환)
denoised_audio = librosa.feature.inverse.mfcc_to_audio(denoised_features.T)

# denoised_audio를 WAV 파일로 저장
output_denoised_audio_file = "denoised_audio.wav"
sf.write(output_denoised_audio_file, denoised_audio, 16000)

print(f"Denoised audio saved to {output_denoised_audio_file}")
