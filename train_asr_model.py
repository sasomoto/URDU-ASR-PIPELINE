from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AdamW

def train_model(dataset_path, model_save_path, epochs=3, lr=5e-5):
    # Load dataset
    dataset = load_dataset("common_voice", "ur", split="train[:5%]")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53", pad_token_id=processor.tokenizer.pad_token_id)

    # Preprocessing
    def preprocess(batch):
        audio = batch["audio"]["array"]
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        with processor.as_target_processor():
            labels = processor(batch["sentence"], padding=True, return_tensors="pt").input_ids
        batch["input_values"] = input_values.squeeze(0)
        batch["labels"] = labels.squeeze(0)
        return batch

    dataset = dataset.map(preprocess)

    # DataLoader
    def collate_fn(batch):
        return {
            "input_values": torch.stack([b["input_values"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

    # Training loop
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataloader:
            input_values = batch["input_values"]
            labels = batch["labels"]
            optimizer.zero_grad()
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    print("Model training completed and saved.")

if __name__ == "__main__":
    train_model("data/processed", "models/final_model")
