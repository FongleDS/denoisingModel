import numpy as np
import librosa
import soundfile as sf
import os


def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def add_gaussian_noise(audio, snr_db):
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    gaussian_noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    noisy_audio = audio + gaussian_noise
    return noisy_audio

def add_white_noise(audio, snr_db):
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    white_noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    noisy_audio = audio + white_noise
    return noisy_audio

def add_external_noise(audio, noise_audio, snr_db):
    noise_audio = noise_audio[:len(audio)]
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    noise_scaling_factor = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
    noisy_audio = audio + noise_audio * noise_scaling_factor
    return noisy_audio

def save_audio(file_path, audio, sr=16000):
    sf.write(file_path, audio, sr)

# 데이터셋 생성
clean_audio_files = [...]  # 깨끗한 음성 파일 리스트
noise_audio_file = 'path_to_noise_audio.wav'
output_dir = 'path_to_output_directory'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

noise_audio = load_audio(noise_audio_file)

for i, clean_file in enumerate(clean_audio_files):
    clean_audio = load_audio(clean_file)

    noisy_gaussian = add_gaussian_noise(clean_audio, snr_db=10)
    noisy_white = add_white_noise(clean_audio, snr_db=10)
    noisy_external = add_external_noise(clean_audio, noise_audio, snr_db=10)

    # 파일 저장
    base_filename = os.path.splitext(os.path.basename(clean_file))[0]

    save_audio(os.path.join(output_dir, f"{base_filename}_gaussian.wav"), noisy_gaussian)
    save_audio(os.path.join(output_dir, f"{base_filename}_white.wav"), noisy_white)
    save_audio(os.path.join(output_dir, f"{base_filename}_external.wav"), noisy_external)

    # 깨끗한 파일도 저장 (필요시)
    # save_audio(os.path.join(output_dir, f"{base_filename}_clean.wav"), clean_audio)

