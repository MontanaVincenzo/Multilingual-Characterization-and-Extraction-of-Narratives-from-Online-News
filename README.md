# Multilingual-Characterization-and-Extraction-of-Narratives-from-Online-News

The internet has opened vast possibilities to easily create direct communication channels between information producers and consumers, potentially leaving the latter exposed to deceptive content and attempts at manipulation. Huge audiences can be affected online, and major crisis events are continuously subjected to the spread of harmful disinformation and propaganda.

In order to foster research and development of novel analytical functionalities to support end-users in analysing the news ecosystem and characterizing manipulation attempts, we launch in the frame of SemEval 2025 campaign a Task on Multilingual Characterization and Extraction of Narratives from Online News.

In particular, the task focuses on the automatic identification of narratives, their classification and identifying the roles of the relevant entities involved. These specific analytical dimensions are of paramount importance for facilitating the work of analysts studying target-specific disinformation phenomena.

The challenge offers three subtasks on news articles: Entity Framing, Narrative Classification and Narrative Extraction, in five languages: Bulgarian, English, Hindi, (European) Portuguese, and Russian. The participants may take part in any number of subtask-language pairs (even just one), and may train their systems using the data for all languages (in a multilingual setup).

We took part to the Entity Framing subtask, making use of a combination of both encoder-only and decoder-only model. More in depth, in this work we use LLaMa 3.2 3B to address the classification task, retrieving properly examples from a training dataset according to the cosine similarities between training and test samples. Vectorization of texts is performed via BERT.
