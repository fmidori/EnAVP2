import argparse
import json

import torch
from torch.optim import RMSprop, Adam, SGD
from Bio import SeqIO

from .model import EnsembleModel as m
from .util import validateSeqs, AVP_PARAMS, AMP_PARAMS


def _optimizer_from_string(string):
    d = {
        "RMSprop" : RMSprop,
        "Adam" : Adam,
        "SGD": SGD
    }
    return d[string]


def load_param_file(fname):
    with open(fname, "r") as handle:
        p = json.load(handle)
    return p


def get_args():
    parser = argparse.ArgumentParser(
        description="Ensemble AntiViral Peptide predictor (training) -- CLI tool")
    parser.add_argument(
        "--pos", dest="pos_file", required=True,
        help=".fasta file containing the positive examples.")
    parser.add_argument(
        "--neg", dest="neg_file", required=True,
        help=".fasta file containing the negative examples.")
    parser.add_argument(
        "--model", dest="output_model_directory", required=True,
        help="Directory where trained model will be stored.")
    parser.add_argument(
        "--ep", dest="n_epochs", default=10, type=int,
        help="Number of epochs to train model for. default=10")
    parser.add_argument(
        "--par", dest="parameter_set", default="avp",
        choices=["avp", "amp", "json"],
        help="Model hyperparameter set, if 'json' must provide --param_file "
             "arg. default=\"avp\"")
    parser.add_argument(
        "--param_file", default=None,
        help="json file with parameter values. "
             "Check documentation for examples."
    )
    parser.add_argument(
        "-c", dest="use_cuda", action="store_true",
        help="if set, will try to use cuda.")
    parser.add_argument(
        "-s", dest="show_results", action="store_true",
        help="if set, will log training progress to stdout.")
    parser.add_argument(
        "--n_batch", default=8, type=int,
        help="Number of sequences in training batch. default=8")
    parser.add_argument(
        "--val", dest="validate", action="store_true",
        help="if set, will perform an extra sanitizing step on the data, replacing "
        "unknown characters with X. If your proteins already contain only standard "
        "aas + X, leave unset.")

    return parser.parse_args()


def run():
    args = get_args()

    if args.parameter_set == "avp":
        p = AVP_PARAMS
    elif args.parameter_set == "amp":
        p = AMP_PARAMS
    else:
        if args.param_file is None:
            print("Need to set --param_file if --par == json. Exiting")
            exit(1)
        p = load_param_file(args.param_file)

    dev = "cuda" if (args.use_cuda and torch.cuda.is_available) else "cpu"

    print(f"Device being utilized: {dev}")

    model = m(
        weight=p["weight"],
        max_depth=p["max_depth"],
        num_layers=p["n_layers"],
        n_estimators=p["n_estimators"],
        embedding_size=p["embedding_size"],
        hidden_layer_size=p["hidden_layer_size"],
        feature_set=p["fset"],
        pseaac_lambda=p.get("pseaac_lambda"),
        pseaac_float=p.get("pseaac_float"),
        dev=dev
    )

    neg_recs = [r for r in SeqIO.parse(open(args.neg_file), "fasta")]
    pos_recs = [r for r in SeqIO.parse(open(args.pos_file), "fasta")]

    if args.validate:
        neg_recs, id_count, _ = validateSeqs(neg_recs, 0, True)
        pos_recs, _, _ = validateSeqs(pos_recs, id_count, True)

    neg_seqs = [r.seq for r in neg_recs]
    pos_seqs = [r.seq for r in pos_recs]


    model.trainDataset(
        pos_seqs=pos_seqs,
        neg_seqs=neg_seqs,
        n_epochs=args.n_epochs,
        lr=p["lr"],
        optimizer=_optimizer_from_string(p["optimizer"]),
        batch_size=args.n_batch,
        show_results=args.show_results)

    model.save(args.output_model_directory)
