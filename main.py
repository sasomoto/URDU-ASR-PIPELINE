import os
import argparse
import logging
from src.utils import setup_logger, ensure_dir_exists
from src.preprocess_audio import preprocess_audio_files
from src.transcription import transcribe_audio
from src.train_model import train_asr_model
from src.evaluate_wer import calculate_wer

# Initialize logger
logger = setup_logger()

# Define main function
def main(args):
    """
    Main function to orchestrate the Urdu ASR pipeline.
    """
    # Ensure output directories exist
    ensure_dir_exists(args.preprocessed_dir)
    ensure_dir_exists(args.models_dir)
    ensure_dir_exists(args.results_dir)
    
    # Step 1: Preprocess Audio
    logger.info("Step 1: Preprocessing Audio Files...")
    preprocess_audio_files(args.input_dir, args.preprocessed_dir, sample_rate=args.sample_rate)

    # Step 2: Train ASR Model
    if args.train:
        logger.info("Step 2: Training ASR Model...")
        train_asr_model(
            preprocessed_dir=args.preprocessed_dir,
            model_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    # Step 3: Transcribe Audio
    logger.info("Step 3: Transcribing Audio Files...")
    transcription_output = os.path.join(args.results_dir, "transcriptions.txt")
    transcribe_audio(
        model_path=os.path.join(args.models_dir, "asr_model.h5"),
        audio_dir=args.preprocessed_dir,
        output_path=transcription_output
    )

    # Step 4: Evaluate WER
    if args.ground_truth:
        logger.info("Step 4: Calculating Word Error Rate (WER)...")
        wer_score = calculate_wer(args.ground_truth, transcription_output)
        logger.info(f"Word Error Rate (WER): {wer_score:.2f}")

    logger.info("Pipeline execution complete.")

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urdu ASR Pipeline")

    # Directories
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw audio files")
    parser.add_argument("--preprocessed_dir", type=str, default="data/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")

    # Training parameters
    parser.add_argument("--train", action="store_true", help="Train the ASR model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    # Evaluation
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth transcriptions for WER evaluation")
    
    # Audio processing
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for audio preprocessing")

    # Parse arguments
    args = parser.parse_args()

    # Run pipeline
    main(args)
