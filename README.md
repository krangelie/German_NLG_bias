# Measuring Gender Bias in German Language Generation

This repository holds code, data, and a German regard classifier as described in our publication in the proceedings of INFORMATIK 2022 ([cam-ready version](https://www.edit.fis.uni-hamburg.de/ws/files/18665970/kraftetal2022_german_regard.pdf)). 
The project is an updated version of the bias measure developed and applied in my Master's thesis ([can be found in this repo](https://github.com/krangelie/bias-in-german-nlg)).


---
## Data

Different development data and experiment artifacts are included in the `data` folder:

For training and evaluation of the classifier:
* `GER_regard_data_annotated/combined_all.csv` contains the majority-labeled and cleaned regard data (`preprocessed_GER/dev_test_indices` contains the randomly determined indices used for our dev and test splits)
* The crowd-sourced human-authored dataset with human annotations (used for training) can be found in: `annotated_data_raw/crowd_sourced_regard_w_annotations`
* The GerPT-2-generated dataset with human annotations (used for classifier evaluation) are in: `annotated_data_raw/gerpt2_generated_regard_w_annotations`

The raw survey data was initially explored with `notebooks/eda.ipynb`. An annotation-ready version was preprocessed with `notebooks/preprocess_raw_survey_data.ipynb`.


Data for experiments:
* `classifier_bias_check` was used to explore classifier-inherent biases
* `gerp2-generated` and `gpt3-generated` contain samples and bias evaluation results with and without triggers
**Warning**: Some samples are explicit or offensive in nature.


---
## Running the code

Switching between modes can be done via python run.py run_mode=MODENAME (`classifier` with `classifier_mode` set to `train` or `evaluate` the classifier, `eval_bias` to run a bias analysis). It is definitely recommended checking out the detailed options within the respective config files. (Python 3.8 recommended)

### Running bias evaluations

The pretrained SentenceBERT-based regard classifier is stored in `models/sbert_regard_classifier.pth`.
To evaluate the bias in a list of sentences:
* Classify the generated sentences with the pretrained regard classifier via `python run.py run_mode=classifier classifier_mode=predict`
* Then run `python run.py run_mode=eval_bias`



### Re-training a regard bias classifier
* Data preprocessing is only needed if you want to train or tune a new classifier. The preprocessed and pre-embedded data from the thesis are also provided with this repository. 
  * Before running the script, make sure to check out `conf/config.yaml` for `dev_settings`, `classifier`, `embedding`, and `preprocessing`. They should be adjusted, depending on the type of classifier you want to train.
  * Preprocess data from the annotated datasets in `data/annotated_data_raw/crowd_sourced_regard_w_annotations` with `run_mode=data`.
  * *Example:* Preparing data for the GRU classifier, can be done as follows: Download [fasttext embeddings](https://www.deepset.ai/german-word-embeddings). Store the `model.bin` in `models/fasttext/`. Then run `python run.py run_mode=data classifier=lstm embedding=fastt pre_processing=for_lstm dev_settings.annotation=majority` (the gru unit type is specified in the classifier settings).


