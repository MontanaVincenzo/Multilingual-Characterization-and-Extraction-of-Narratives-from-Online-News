import csv
from sklearn import metrics
import numpy as np
import argparse
import logging
import sys

logger = logging.getLogger("semeval2024_t10_st2_scorer")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def evaluate(gold, predictions, classes):
    """
    Evaluate the performance of the classifier on the test data.
    """
    f1_scores_per_instance = []
    for true_labels, predicted_labels in zip(gold, predictions):
        # One-hot encode labels
        true_labels_onehot = np.zeros(len(classes), dtype=int)
        for label in true_labels:
            if label in classes:
                true_labels_onehot[classes.index(label)] = 1
        predicted_labels_onehot = np.zeros(len(classes), dtype=int)
        for label in predicted_labels:
            if label in classes:
                predicted_labels_onehot[classes.index(label)] = 1
        f1_score = metrics.f1_score(true_labels_onehot, predicted_labels_onehot, zero_division=0)
        f1_scores_per_instance.append(f1_score)
    f1_scores = np.array(f1_scores_per_instance)
    return f1_scores.mean(), f1_scores.std()


def read_and_evaluate(prediction_file, gold_file, classes_file_coarse, classes_file_fine):
    """
    Evaluates predictions from a TSV file against a gold standard TSV file.

    Args:
        prediction_file (str): Path to the TSV file with predictions (ID, NARRATIVE, SUBNARRATIVE).
        gold_file (str): Path to the TSV file with gold standard labels (ID, NARRATIVE, SUBNARRATIVE).
        classes_file_coarse (str): Path to the coarse (narrative) classes
        classes_file_fine (str): Path to the fine (subnarrative) classes
    Returns:
        F1 mean and standard deviation
    """

    # Read the classes
    with open(classes_file_coarse, 'r') as f:
        classes_coarse = f.read().split('\n')
    with open(classes_file_fine, 'r') as f:
        classes_fine = f.read().split('\n')

    # Read predictions and gold labels
    coarse_predictions, fine_predictions = {}, {}
    with open(prediction_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        #next(reader)  # Skip header - uncomment if there is a header
        for row in reader:
            coarse_predictions[row[0]] = row[1].split(';')  # Split predictions into lists
            fine_predictions[row[0]] = row[2].split(';')  # Split predictions into lists

    gold_labels_coarse, gold_labels_fine = {}, {}
    with open(gold_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        # next(reader)  # Skip header - uncomment if there is a header
        for row in reader:
            gold_labels_coarse[row[0]] = row[1].split(';')  # Split gold labels into lists
            gold_labels_fine[row[0]] = row[2].split(';')  # Split gold labels into lists

    # Ensure same IDs in both files
    common_ids_coarse = set(coarse_predictions.keys()) & set(gold_labels_coarse.keys())
    common_ids_fine = set(fine_predictions.keys()) & set(gold_labels_fine.keys())
    coarse_misalignment = 2*len(common_ids_coarse) != len(coarse_predictions)+len(gold_labels_coarse)
    fine_misalignment = 2*len(common_ids_fine) != len(fine_predictions)+len(gold_labels_fine)
    if fine_misalignment or coarse_misalignment:
        error_message = f"ERROR: Mismatch in IDs between prediction and gold files!"
        error_message += (f'\n#coarse: {len(coarse_predictions)}, #fine: {len(fine_predictions)}, #gold coarse: {len(gold_labels_coarse)}, #gold fine: {len(gold_labels_fine)}')
        error_message += '\nMistaken narrative predicted IDs ==> ' + ', '.join(set(coarse_predictions.keys()) - set(gold_labels_coarse.keys()))
        error_message += '\nMistaken subnarrative predicted IDs ==> ' + ', '.join(set(fine_predictions.keys()) - set(gold_labels_fine.keys()))
        logger.error(error_message)
        raise ValueError(error_message)

    # Flatten lists for sklearn metrics
    y_true_fine = [gold_labels_fine[g] for g in gold_labels_fine]
    y_true_coarse = [gold_labels_coarse[g] for g in gold_labels_coarse]
    y_pred_fine = [fine_predictions[p] for p in fine_predictions]
    y_pred_coarse = [coarse_predictions[p] for p in coarse_predictions]

    # Calculate metrics
    f1_c, f1_sd_c = evaluate(y_true_coarse, y_pred_coarse, classes_coarse)
    f1_f, f1_sd_f = evaluate(y_true_fine, y_pred_fine, classes_fine)
    return f1_c, f1_sd_c, f1_f, f1_sd_f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', '-p', type=str, required=True, help='Path to the prediction file.')
    parser.add_argument('--gold_file', '-g', type=str, required=True, help='Path to the gold file.')
    parser.add_argument('--classes_file_fine', '-f', type=str, required=True, help='Path to the file with the fine-grained classes.')
    parser.add_argument('--classes_file_coarse', '-c', type=str, required=True, help='Path to the file with the coarse-grained classes')
    parser.add_argument("--log_to_file", "-l", action='store_true', default=False, help="Set flag if you want to log the execution to a file.")
    args = parser.parse_args()

    if args.log_to_file:
        output_log_file = args.prediction_file + ".log"
        logger.info("Logging execution to file " + output_log_file)
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)
        logger.setLevel(logging.DEBUG)

    f1_c, f1_sd_c, f1_f, f1_sd_f = read_and_evaluate(args.prediction_file, args.gold_file, args.classes_file_coarse, args.classes_file_fine)

    if not args.log_to_file:
        print(f"Evaluation Results:")
        print(f"F1@coarse: {f1_c:.3f} ({f1_sd_c:.3f})")
        print(f"F1@fine: {f1_f:.3f} ({f1_sd_f:.3f})")
    else:
        print(f"{f1_c:.3f}\t{f1_sd_c:.3f}\t{f1_f:.3f}\t{f1_sd_f:.3f}")
        logger.info(f"Evaluation Results:\nF1@coarse: {f1_c:.3f} ({f1_sd_c:.3f})\nF1@fine: {f1_f:.3f} ({f1_sd_f:.3f})")

