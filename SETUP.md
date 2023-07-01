# aci-demo-benchmark-private


This is the code base for running the baselines in our paper:
{CITATION}


## Setup

Please go through the following steps for setup.

1. Download data

Please download from figshare link:
[FINAL DOI LINK TO BE ADDED HERE]

Then please move the data into the project directory under as data/

2. UMLS resource setup

We use QuickUMLS (https://github.com/Georgetown-IR-Lab/QuickUMLS) for one of our medical fact based evluations and retrieval-based lines. In order to use this you will need to setup a UMLS account (https://uts.nlm.nih.gov/uts/).

- Download the umls 2022AA metathesaurus version
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html
- Unzip this file and move it under resources/ resulting in a resources/

Please move your {MRCONSO,MRSTY}.RRF files to resources/ folder.
Make sure your install is in the resources/des folder.
```
python -m quickumls.install resources/ resources/des
```


3. Package installation

Installation method 1:
```
pip install -r requirements.txt
```

Installation method 2:
```
conda create -n py37_acidemo python=3.7.15 anaconda
conda activate py37_acidemo

pip install quickumls
pip install evaluate==0.4.0
pip install nltk==3.8.1
pip install rouge-score==0.1.2
pip install bert-score==0.3.12
pip install git+https://github.com/google-research/bleurt.git
pip install spacy==3.4.4
pip install scispacy==0.5.1
pip install accelerate==0.15.0
pip install datasets==2.9.0
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.5.0/en_core_web_md-3.5.0-py3-none-any.whl
```


## Expected data and folder structure setup

After setup your folder structure should be as follows:

- data/
    - src_experiment_data/
        - {train,valid,clinicalnlptaskB_test1,clinicalnlp_taskC_test2,clef_taskC_test3}.csv
        - {train,valid,clinicalnlptaskB_test1,clinicalnlp_taskC_test2,clef_taskC_test3}_metadata.csv
    - challenge_data/
        - {train_aci,valid_aci,test1_aci,test2_aci,test3_aci}_{asrcorr,asr}.csv
        - {train_aci,valid_aci,test1_aci,test2_aci,test3_aci}_{asrcorr,asr}_metadata.csv
        - {train_virtscribe,valid_virtscribe,test1_virtscribe,test2_virtscribe,test3_virtscribe}_{humantrans,asr}.csv
        - {train_virtscribe,valid_virtscribe,test1_virtscribe,test2_virtscribe,test3_virtscribe}_{humantrans,asr}_metadata.csv

- resources/
    - MRCONSO.RRF
    - MRSTY.RRF
    - semantic_types.txt

- baselines/
    - bart_summarization.py - python code for running bart models. This code is adapted from [BioBART](https://github.com/GanjinZero/BioBART).
    - bart-LED_bashed.sh - shell script to run experimental settings from paper
    - baseline_transcript_retreival.ipynb - notebook for running the transcript copy/retrieval baselines
    - longformer_summarization.py - python code from running longformer encoder-decoder (LED) model
    - post-process.ipynb - used for reformatting the json to csv files and generating a bash evaluaion script that calls the evaluate_summarizaion.py file
    - pre-process.ipynb - used for reformatting the csv to a json file
    - sectiontagger.py - code for rule-based section/division detector
    - semantics.py - used fo identifyin umls semantic types

- evaluation/
    - data_statistics.py - script used to generate data statistics
    - evaluate_summarization.py - python evaluation scipt
    - UMLS_evaluation.py - used for identifyin medical named entities


## Running the baselines

1. Preprocess data into json files: 

Please run `baselines/pre-process.ipynb`

2. Run baselines: 

transcript and IR baselines:
```
baselines/bart_transcript_retrieval.ipynb 
```

bart/LED baselines:
```
python baselines/sectiontagger.py 
bash baselines/bart-LED_based.sh data
```

As a result, the experiment directory will be populated with *.json output files.


3. Postprocess data into csv files and generate shell scripts for running the evaluations

Please run:
```
baselines/post-process.ipynb
```
At the end of an evaluation_script.sh file

4. Evaluation

Run the generated evaluation from the previous step.
```
./baselines/evaluation_script.sh
```
