import argparse
import warnings
import torch
import sys
import os
from Bio import SeqIO

from .model import EnsembleModel as m
from .util import IUPAC, nextn, validateSeqs

MODEL_DIR="models/"

def get_args():
    parser = argparse.ArgumentParser(
        description="Ensemble AntiViral Peptide predictor -- CLI tool")
    parser.add_argument(
        "-o", dest="output_file", default="enavp_predictions.tsv",
        help="File where predictions will be stored. "
        "Can be set to stdout by passing \"stdout\". "
        "default=enavp_predictions.tsv")
    parser.add_argument(
        "-f", dest="output_format",
        default="tsv", choices=["tsv", "csv"],
        help="Output format, either \"tsv\" or \"csv\". default=\"tsv\"")
    parser.add_argument(
        "-m",dest="model_directory",
        help="Directory to load model from.", default='src/enavp/pre_trained_models/AVP/full_ds')
    parser.add_argument(
        "-c",dest="use_cuda", action="store_true",
        help="If set, will try to use cuda.")
    parser.add_argument(
        "--n_batch", default=100,
        help="Number of sequences to load in memory at a time. Increase for "
             "faster runtime and decrease for less memory usage. default=100")
    parser.add_argument(
        "input_file",
        help=".fasta file with protein sequences to be evaluated.")

    return parser.parse_args()


def evaluateSequences(model, parser, out_handle, n_batch, sep):
    i = 1
    warned = False
    with torch.no_grad():
        for record_list in nextn(parser,n_batch):
            record_list, i, warned = validateSeqs(record_list,i, warned)
            pred = model.predict([r.seq for r in record_list])
            for r,p in zip(record_list, pred):
                out_handle.write(f"{r.id}{sep}{p.item():.5f}\n")


def run():
    # parse user input
    args = get_args()

    sep = "," if args.output_format == ".csv" else "\t"

    model_dir = args.model_directory

    dev = "cuda" if (args.use_cuda and torch.cuda.is_available) else "cpu"

    try:
        in_handle = open(args.input_file)
    except FileNotFoundError:
        print(f"File {args.input_file} not found.\n Exiting.")
        exit()

    parser = SeqIO.parse(in_handle, "fasta")

    out_handle = open(args.output_file, "w") if args.output_file != "stdout" else sys.stdout

    model = m.load_from(model_dir, dev)

    out_handle.write(f"ID{sep}SCORE\n")
    evaluateSequences(model, parser, out_handle, args.n_batch, sep)

    print("Finished predictions")
    in_handle.close()
    out_handle.close()
