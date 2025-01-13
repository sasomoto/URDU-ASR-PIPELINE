import os

def get_file_paths(directory, extension=".wav"):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

def create_directory_structure(base_dir):
    directories = [
        "data/raw_audio", "data/transcriptions", "data/processed", "data/test_samples",
        "models/checkpoints", "models/final_model", "models/tokenizer",
        "results/logs", "results/plots",
    ]
    for dir_name in directories:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)

if __name__ == "__main__":
    create_directory_structure(".")
