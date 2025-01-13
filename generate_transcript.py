import os
from transformers import pipeline

def transcribe_audio(input_dir, output_file):
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.wav'):
                input_path = os.path.join(input_dir, file_name)
                try:
                    result = transcriber(input_path)
                    f.write(f"{file_name}\t{result['text']}\n")
                    print(f"Transcribed: {file_name}")
                except Exception as e:
                    print(f"Error transcribing {file_name}: {e}")

if __name__ == "__main__":
    transcribe_audio("data/processed", "data/transcriptions/transcriptions.txt")
