# EnAVP

Ensemble AntiViral Peptide prediction tool.

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

## Using

To run the model on a fasta file:

    enavp <input.fasta> --out out.tsv

Use `enavp -h` to see aditional documentation and options.

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
