import os
import librosa
import soundfile as sf

def preprocess_audio(input_dir, output_dir, target_sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            try:
                audio, sr = librosa.load(input_path, sr=target_sr)
                sf.write(output_path, audio, target_sr)
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    preprocess_audio("data/raw_audio", "data/processed")
