import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
import soundfile as sf
import os
import time

# 경로 설정
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
print(len(cleandata))
noisedata = os.listdir(noisedir)
print(noisedata)
print(len(noisedata))

datalen = min(len(noisedata), len(cleandata))
# print(f"Using {datalen} files from each directory.")

timeslice_sec = 5


# 오디오 로드 함수
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


# 오디오 분할 함수
def split_audio(audio, sr=16000, duration=timeslice_sec):
    length = sr * duration
    chunks = [audio[i:i + length] for i in range(0, len(audio), length) if len(audio[i:i + length]) == length]
    return chunks


# 오토인코더 모델 빌드 함수
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


# 파일 경로 리스트
# clean_audio_files = [os.path.join(cleandir, f) for f in cleandata[:datalen]]
# print(len(clean_audio_files))
# noise_audio_files = [os.path.join(noisedir, f) for f in noisedata[:datalen]]
# print(len(noise_audio_files))


clean_audio_files = [os.path.join(cleandir, f) for f in cleandata]
print(len(clean_audio_files))
noise_audio_files = [os.path.join(noisedir, f) for f in noisedata]
print(len(noise_audio_files))

# 청크 리스트 초기화
clean_chunks = []
noise_chunks = []

for clean_file in clean_audio_files:
    clean_audio = load_audio(clean_file)
    clean_split = split_audio(clean_audio)
    # clean_chunks.extend(clean_split)
    print(f"Processed {clean_file}, {len(clean_split)} chunks")
    if len(clean_split) > 0:
        clean_chunks.extend(clean_split)
        # print(f"Processed {clean_file}, {len(clean_split)} chunks")
    else:
        print(f"Skipped {clean_file}, no valid chunks")

# 청크 분할
for noise_file in noise_audio_files:
    noise_audio = load_audio(noise_file)
    noise_split = split_audio(noise_audio)
    # noise_chunks.extend(noise_split)
    print(f"Processed {noise_file}, {len(noise_split)} chunks")
    if len(noise_split) > 0:
        noise_chunks.extend(noise_split)
        # print(f"Processed {noise_file}, {len(noise_split)} chunks")
    else:
        print(f"Skipped {noise_file}, no valid chunks")

# 청크 형태 변환
clean_chunks = [np.expand_dims(audio, axis=-1) for audio in clean_chunks]  # (length, 1) 형태로 변환
print(len(clean_chunks))
noise_chunks = [np.expand_dims(audio, axis=-1) for audio in noise_chunks]  # (length, 1) 형태로 변환
print(len(noise_chunks))

# 길이를 맞추기 위해 짧은 길이를 기준으로 슬라이스
min_len = min(len(noise_chunks), len(clean_chunks))
print("Min len: ", min_len)
X_train = np.array(noise_chunks[:min_len])
y_train = np.array(clean_chunks[:min_len])
print(len(X_train), "and", len(y_train))

assert X_train.shape[1] == y_train.shape[1], f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
print(f"Training samples: {len(X_train)} pairs")

# 모델 정의
input_shape = (16000 * timeslice_sec, 1)  # 5초 길이의 오디오 신호 (16000Hz 샘플링 레이트)
autoencoder = build_autoencoder(input_shape)
autoencoder.summary()

# 모델 학습 시간 측정
start_time = time.time()

# 모델 학습
history = autoencoder.fit(X_train, y_train, epochs=2, batch_size=32, shuffle=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f} seconds")

# 모델 학습
# autoencoder.fit(X_train, y_train, epochs=2, batch_size=32, shuffle=True)

# 모델 저장
model_path = "./model/denoising_autoencoder.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
autoencoder.save(model_path)

# 테스트 데이터에 대한 소음 제거 및 저장
test_noise_audio_file = "./test_original/test_noisy.wav"

# 테스트 데이터 로드 및 노이즈 추가
test_noise_audio = load_audio(test_noise_audio_file)
test_noisy_audio = np.expand_dims(test_noise_audio[:16000 * timeslice_sec], axis=-1)  # 5초 길이로 자르고 (length, 1) 형태로 변환

# 모델 로드
autoencoder = tf.keras.models.load_model(model_path)

# 소음 제거 예측
denoised_audio = autoencoder.predict(np.expand_dims(test_noisy_audio, axis=0))
denoised_audio = np.squeeze(denoised_audio, axis=0)

# denoised_audio를 WAV 파일로 저장
output_denoised_audio_file = "./denoisedTestset/denoised_audio.wav"
os.makedirs(os.path.dirname(output_denoised_audio_file), exist_ok=True)
sf.write(output_denoised_audio_file, denoised_audio, 16000)

print(f"Denoised audio saved to {output_denoised_audio_file}")
