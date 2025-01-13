import csv
from sklearn import metrics
import numpy as np
import argparse


class T10ST2RandomBaseline:

    def __init__(self, classes_coarse, classes_fine):
        """
        Initialize the random classifier with a list of classes.
        """
        self.classes_coarse = classes_coarse
        self.classes_fine = classes_fine

    def predict(self, texts):
        """
        Assign random labels to each instance in the test data.
        """
        predictions_coarse = [np.random.choice(self.classes_coarse, np.random.randint(1, 2)) for _ in texts]
        predictions_fine = [np.random.choice(self.classes_fine, np.random.randint(1, 3)) for _ in texts]
        return predictions_coarse, predictions_fine

    def predict_and_write(self, texts, ids, filename):
        """
        Predict labels for the test data and write the predictions to a file.
        """
        predictions_coarse, predictions_fine = self.predict(texts)
        # write to file
        with open(filename, 'w', newline='') as coarse_file:
            tsv_writer = csv.writer(coarse_file, delimiter='\t')
            # tsv_writer.writerow(['ID', 'NARRATIVE', 'SUBNARRATIVE'])  # uncomment for a header
            for doc_id, coarse_pred, fine_pred in zip(ids, predictions_coarse, predictions_fine):
                tsv_writer.writerow([doc_id, ";".join(coarse_pred), ";".join(fine_pred)])


if __name__ == '__main__':
    """
    >> python subtask2_baseline_random.py -i 'subtask-2-annotations.txt' -f subtask2_subnarratives.txt -c subtask2_narratives.txt
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', '-i', type=str, required=True, help='Path to the test file.')
    parser.add_argument('--class_file_fine', '-f', type=str, required=True, help='Path to the fine-grained classes.')
    parser.add_argument('--class_file_coarse', '-c', type=str, required=True, help='Path to the coarse-grained classes')
    args = parser.parse_args()

    # Read the classes
    with open(args.class_file_fine, 'r') as f:
        classes_fine = f.read().split('\n')
    with open(args.class_file_coarse, 'r') as f:
        classes_coarse = f.read().split('\n')

    test_x = {}
    with open(args.test_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        #next(reader)  # Skip header
        for row in reader:
            test_x[row[0]] = row[1]

    # write out
    random_baseline = T10ST2RandomBaseline(classes_coarse, classes_fine)
    random_baseline.predict_and_write(test_x.values(), test_x.keys(), f'subtask2_baselines/st2_random_predictions.txt')
