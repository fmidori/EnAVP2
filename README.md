# EnAVPCLass
http://cetics.butantan.gov.br/ceticsdb/classificationtools/enavpclass/ (online version in construction)
Antiviral Peptides prediction and classification. 
This repo is a draft for yutaka/EnAVPClass, oficial version of EnAVPClass. 
EnAVPClass is a two-stage classifier method for the prediction and classification of AVPs. The first stage consists of an ensemble model of a Deep Recurrent Neural Network and a Random Forest module, built to predict the antiviral activity of unknown peptides. The second stage uses a Support Vector Machine to classify the AVPs by their mechanisms of action, classified in (i) Membrane (AVPs that interact with the viral capsid membrane), (ii) Replication (AVPs that interfere with viral replication inside the cell), and (iii) Assembly (AVPs that interfere with the viral assembly of new particles inside the cell). 

## Instalation

Before installing is recomended you create a virtual environment (python 3.8 or 3.9).
For example, using python's `venv`:

    git clone <repo url>
    cd EnAVP
    python -m venv enavp_env
    source enavp_env/bin/activate

Install all dependencies in new environment:

    pip install -r requirements.txt

Install EnAVP (will only instal in current environment):

    python setup.py install

## Using for Prediction

To run the model on a fasta file:

    enavp <input.fasta> --out out.tsv

Use `enavp -h` to see aditional documentation and options.

## Using for Prediction and Classification 

To run the model on a fasta file:

    enavp-classify <input.fasta> --out out.tsv


## Tests

To run the tests described in \[PAPER TK\]

    enavp-test

## Training with Custom Data

To train with your own data:

    enavp-train <positive_class.fasta> <negative_class.fasta> --model <model_dir>

You can then use the `--model` flag to select the new model:

    enavp <input.fasta> --out out.tsv --model <model_dir>

This is only recommended if you wish to validate the results with an updated
dataset, to predict sequences in a different context you should probably only take
the code as inspirations and tweak the hyperparameters and data loading as required.

Use `enavp-train -h` to check aditional options.
