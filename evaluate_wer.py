from jiwer import wer

def evaluate_wer(ground_truth_file, predictions_file):
    with open(ground_truth_file, 'r', encoding='utf-8') as gt_file, \
         open(predictions_file, 'r', encoding='utf-8') as pred_file:
        ground_truths = [line.strip().split('\t')[1] for line in gt_file]
        predictions = [line.strip().split('\t')[1] for line in pred_file]

    error_rate = wer(ground_truths, predictions)
    print(f"Word Error Rate (WER): {error_rate * 100:.2f}%")

if __name__ == "__main__":
    evaluate_wer("data/transcriptions/ground_truth.txt", "data/transcriptions/predictions.txt")
