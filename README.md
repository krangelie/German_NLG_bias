# Measuring Gender Bias in German Language Generation

This repository holds code, data, and a German regard classifier as described in our [publication](https://dl.gi.de/handle/20.500.12116/39481) in the proceedings of INFORMATIK 2022. 
The project is an updated version of the bias measure developed and applied in my Master's thesis ([can be found in this repo](https://github.com/krangelie/bias-in-german-nlg)).


---
## Running bias evaluations

(Python 3.7 recommended)

The pretrained SentenceBERT-based regard classifier is stored in `models/sbert_regard_classifier.pth`.
To evaluate the bias in a list of sentences:
* Classify the generated sentences with the pretrained regard classifier via `python run.py run_mode=classifier classifier_mode=predict`
* Then run `python run.py run_mode=eval_bias`

**Generally:** 
Switching between modes can be done via python run.py run_mode=MODENAME (`classifier` with 
`classifier_mode` set to `train` or `evaluate` the classifier, `eval_bias` to run a bias analysis). It is definitely recommended checking out the detailed options within the respective config files. 

---
## Data

Different development data and experiment artifacts are included in the `data` folder:

For training and evaluation of the classifier:
* `GER_regard_data_annotated/combined_all.csv` contains the majority-labeled and cleaned regard data (`preprocessed_GER/dev_test_indices` contains the randomly determined indices used for our dev and test splits)
* The crowd-sourced human-authored dataset with human annotations (used for training) can be found in: `annotated_data_raw/crowd_sourced_regard_w_annotations`
* The GerPT-2-generated dataset with human annotations (used for classifier evaluation) are in: `annotated_data_raw/gerpt2_generated_regard_w_annotations`

Jupyter notebooks used for data & annotation exploration can be found in the original thesis repo.

Data for experiments:
* `classifier_bias_check` was used to confirm that there was no classifier-inherent biases
* `gerp2-generated` and `gpt3-generated` contain samples and bias evaluation results with and without triggers
**Warning**: Some samples are explicit or offensive in nature.
