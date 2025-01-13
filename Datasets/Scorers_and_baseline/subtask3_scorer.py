import logging.handlers
import argparse
import os
import csv
import sys
from evaluate import load

"""
Requirments:
!pip install evaluate
!pip install bert_score
"""


"""
Scoring of SEMEVAL-Task-10--subtask-3 with the bertscore f1
"""

sys.path.append('data')
bertscore = load("bertscore")

logger = logging.getLogger("task10-3_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('data/app.log')
file_handler.setFormatter(formatter)
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

logger.addHandler(file_handler)

def _read_gold_and_pred(pred_fpath, gold_fpath):
    """
    Read gold and predicted data.
    :param pred_fpath: a tsv file with predictions,
    :param gold_fpath: the original gold file with the explanations.
    :return: {id:pred_labels} dict; {id:gold_labels} dict
    """

    gold_labels = {}
    with open(gold_fpath, encoding="utf-8", ) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            id, main_narr,sub_narr,expl = row
            gold_labels.update({id: expl})

    pred_labels = {}
    with open(pred_fpath, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            id, expl = row
            pred_labels.update({id: expl})

    if set(gold_labels.keys()) != set(pred_labels.keys()):
        logger.error(
            'There are either missing or added examples to the prediction file. Make sure you only have the gold examples in the prediction file.')
        raise ValueError(
            'There are either missing or added examples to the prediction file. Make sure you only have the gold examples in the prediction file.')

    return pred_labels, gold_labels


def evaluate(pred_fpath, gold_fpath):
    """
      Evaluates the generated explanation w.r.t. a gold standard explanations.
      Metrics are: precision,recall and f1 derived from bert-score
      :param pred_fpath: a txt file with generated explanation,
      :param gold_fpath: the original txt gold file.

    """
    pred_explanations, gold_explanations = _read_gold_and_pred(pred_fpath, gold_fpath)

    #lang_labels = {}

    #with open(lang_fpath, encoding="utf-8") as file:
    #    reader = csv.reader(file, delimiter='\t')

    #    for row in reader:
    #        id, lang = row
    #        if lang not in lang_labels.keys():
    #            lang_labels[lang] = list()
    #        lang_labels[lang].append(id)

    results = {"precision": [], "recall": [], "f1": []}
    #for lang in lang_labels.keys():
    #lang_ids = list(lang_labels[lang])
    predictions = [v for k, v in pred_explanations.items()]
    references = [v for k, v in gold_explanations.items()]
    lang_results = bertscore.compute(predictions=predictions, references=references, model_type="bert-base-multilingual-cased")
    results["precision"] = results["precision"] + lang_results["precision"]
    results["recall"] = results["recall"] + lang_results["recall"]
    results["f1"] = results["f1"] + lang_results["f1"]
    agg_results = dict()

    for k in ["precision","recall","f1"]:
        try:
            agg_results[k] = sum(results[k]) / len(results[k])
        except Exception as e:
            logger.errors(f"Results could not be computed probably division by 0: {e}")
    agg_results["hashcode"]=lang_results["hashcode"]

    # {'precision': 0.666784530878067,
    # 'recall': 0.6953270077705384,
    # 'f1': 0.6806213170289993}
    return list(agg_results.values())




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file_path",
        '-g',
        type=str,
        required=True,
        help="Paths to the file with the gold annotations."
    )
    parser.add_argument(
        "--pred_file_path",
        '-p',
        type=str,
        required=True,
        help="Path to the file with predictions"
    )
    parser.add_argument(
        "--log_to_file",
        "-l",
        action='store_true',
        default=False,
        help="Set flag if you want to log the execution file. The log will be appended to <pred_file>.log"
    )

    args = parser.parse_args()

    pred_file = args.pred_file_path
    gold_file = args.gold_file_path
    if args.log_to_file:
        output_log_file = pred_file + ".log"
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)
        logger.setLevel(logging.DEBUG)  #
        logger.info("Logging execution to file " + output_log_file)
    else:
        logger.addHandler(ch)

    if args.log_to_file:
        logger.info('Reading gold file')
    else:
        logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
    if args.log_to_file:
        logger.info('Reading predictions file')
    else:
        logger.info('Reading predictions file {}'.format(args.pred_file_path))


    precision, recall, f1,model_hash = evaluate(pred_file, gold_file)
    logger.info("precision={:.5f}\trecall={:.5f}\tf1={:.5f}".format(precision, recall, f1))
    if args.log_to_file:
        print("{}\t{}\t{}".format(precision, recall, f1))




'''
USAGE EXAMPLE:
python3 subtask3_scorer.py -g dev/BG/subtask-3-annotations.txt -p subtask3_baseline/BG/baseline_phi.txt
python3 subtask3_scorer.py -g dev/PT/subtask-3-annotations.txt -p subtask3_baseline/PT/baseline_phi.txt
'''