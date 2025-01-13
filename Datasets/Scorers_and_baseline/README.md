# Data for SemEval 2025 task 10

This Readme is distributed with the data for participating in SemEval 2025 task 10. 
The website of the shared task (https://propaganda.math.unipd.it/semeval2025task10/), includes a detailed descripton of the tasks, the submission instructions, updates on the competition and a live leaderboard.


## Task Description

### Subtask 1: Entity Framing

Given a news article and a list of mentions of named entities (NEs) in the article, assign for each such mention one or more roles using a predefined taxonomy of fine-grained roles covering three main type of roles: protagonists, antagonists, and innocent. This is a multi-label multi-class text-span classification task.

### Subtask 2: Narrative Classification

Given a news article and a two-level taxonomy of narrative labels (where each narrative is subdivided into subnarratives) from a particular domain, assign to the article all the appropriate subnarrative labels. This is a multi-label document classification task.

### Subtask 3: Narrative Extraction
Given a news article and a dominant narrative of the text of this article, generate a free-text explanation (up to max. of 80 words) supporting the choice of this dominant narrative. The to-be-generated explanation should be grounded in the text fragments that provide evidence of the claims of the dominant narrative. This is a text-to-text generation task.


## Data Format 

The format of the input, gold label and prediction files is specified on the [website of the competition](https://propaganda.math.unipd.it/semeval2025task10/). Note that the scorer will only accept files with the .txt extension.


## Baselines and Scorers 

### Subtask 1

#### Baseline Code Usage

For subtask 1, we generate 2 baselines for each language: a random guessing baseline, and a majority voting baseline. To generate baselines for all four languages (BG, EN, HI, PT):

```bash
# For BG (Bulgarian)
python subtask1_baseline.py --dev_file data/BG/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/BG --baseline_type random
python subtask1_baseline.py --train_file data/BG/subtask-1-annotations.txt --dev_file data/BG/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/BG --baseline_type majority

# For EN (English)
python subtask1_baseline.py --dev_file data/EN/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/EN --baseline_type random
python subtask1_baseline.py --train_file data/EN/subtask-1-annotations.txt --dev_file data/EN/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/EN --baseline_type majority

# For HI (Hindi)
python subtask1_baseline.py --dev_file data/HI/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/HI --baseline_type random
python subtask1_baseline.py --train_file data/HI/subtask-1-annotations.txt --dev_file data/HI/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/HI --baseline_type majority

# For PT (Portuguese)
python subtask1_baseline.py --dev_file data/PT/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/PT --baseline_type random
python subtask1_baseline.py --train_file data/PT/subtask-1-annotations.txt --dev_file data/PT/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/PT --baseline_type majority

```

The predictions are saved as baseline_random.txt and baseline_majority.txt in the specified output directories under subtask1_baselines.

#### Scorer Code Usage
The subtask1_scorer.py script evaluates the baselines by comparing predictions against the gold labels. 


<!-- Use the following commands to score both the random and majority baselines for each language. The **-g** argument is for the ground truth file path and **-p** is for the predictions file path -->

```bash
python subtask1_scorer.py -g data/BG/subtask-1-entity-mentions.txt -p data/BG/subtask-1-entity-mentions.txt 
```
Using the command above for example, you should receive 1.00 on all metrics. The **-g** argument is for the ground truth file path and **-p** is for the predictions file path.

| EMR (*) | Micro Precision | Micro Recall | Micro F1 | Main Role Accuracy |
|---------|-----------------|--------------|----------|---------------------|
| 1.0000  | 1.0000          | 1.0000       | 1.0000   | 1.0000             |

(*) EMR is the Exact Match Ratio on the fine-grained roles which is the official evaluation metric.

#### Format Checker

The scorer already checks the format and provides logs in case of errors. For those interested in using the format checker only without the scorer, below is an example of usage in python:


```python
from subtask1_scorer import MAIN_ROLES, FINE_GRAINED_ROLES, read_file, check_file_format

gold_file_path = "data/train/BG/subtask-1-annotations.txt"
pred_file_path = "data/train/BG/subtask-1-annotations.txt"

gold_dict = read_file(gold_file_path)
pred_dict = read_file(pred_file_path)

format_errors = check_file_format(gold_dict, pred_dict)
```

### Subtask 2

For subtask 2, we release a random baseline, requiring the path to the evaluation test instances (**--test_file, -i**), the path to the (fine-grained) subnarrative classes (**--class_file_fine, -f**), and the path to the (coarse-grained) narrative classes (**--class_file_coarse, -c**).

```
>> python subtask2_baseline_random.py -i subtask-2-annotations.txt -f subtask2_subnarratives.txt -c subtask2_narratives.txt
```

The scorer will can run as follows, with **-g** referring to the ground truth filepath, **-p** to the predictions filepath, **-f** to the (fine-grained) subnarrative classes, **-c** to the (coarse-grained) narrative classes.

```
>> python subtask2_scorer.py -g subtask-2-annotations.tsv -p t10-st2-predictions-fine.tsv -f subtask2_subnarratives.txt -c subtask2_narratives.txt

Evaluation Results:
F1@coarse: 0.020 (0.089)
F1@fine: 0.007 (0.045)

```

### Subtask 3


The baseline model used in this task will be the [Phi3-mini](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) with a 8k context length in tokens and 7B
parameter 4. The prompt used to generate the texts in all languages is the following:

>Given a news article along with its dominant and sub-dominant narratives, generate a concise text (maximum 80 words) supporting these narratives without the need to explicitly mentioning them. The explanation should align with the language of the article and be direct and to the point. If the sub-
dominant narrative is ’Other,’ focus solely on supporting the dominant narrative. The response should be clear, succinct, and avoid unnecessary elaboration.
>
>Dominant Narrative: (dominant narrative class) 
>
>Sub-dominant Narrative:(sub-dominant narrative class)
>
>Article: (article text)

The output of the language model is then processed to ensure that the limit of 80 words is respected.
For that, we will rely on word separation using white spaces. If the limit of 80 words is surpassed,
then the text will be truncated at the end to the limit of 80 words

Below is an example of how to get a baseline of the subtask3. The baseline assumes that the file narratives_gt (which is the file with the ground-truth for the dominant and sub-dominant narratives) is in the path data/out_examples/". The script also assumes that the file for the raw-articles are in "data/out_examples/raw-articles". This parameters can be changed inside the script in the getArticlesText function using the npath and apath arguments, respectively.

```
python subtask3_baseline.py 
```

Below is a usage example of the scorer function on the  baseline. The **-g** argument is for the ground truth file path and **-p** is for the predictions file path. 

```
python3 subtask3_scorer.py -g dev/BG/subtask-3-annotations.txt -p subtask3_baseline/BG/baseline_phi.txt

```


