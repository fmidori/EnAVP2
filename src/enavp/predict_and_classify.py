import argparse
import warnings
import torch
import sys
import os
from Bio import SeqIO

from .model import EnsembleModel as m
from .util import IUPAC, nextn, validateSeqs

import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .preprocess import merge_features,preprocess_for_model
from .features import PAAC,DPC,readFasta

#model_dir="src/enavp/pre_trained_models/AVP/full_ds"
clf = joblib.load("src/enavp/pre_trained_models/ClassAVP/model.pkl")
scaler = joblib.load("src/enavp/pre_trained_models/ClassAVP/std_scaler.pkl")

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
        help="Directory to load model from.",default='src/enavp/pre_trained_models/AVP/full_ds')
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


def evaluateSequences(model, parser, out_handle, n_batch, sep, classif, proba):
    i = 1
    warned = False
    with torch.no_grad():
        for record_list in nextn(parser,n_batch):
            record_list, i, warned = validateSeqs(record_list,i, warned)
            pred = model.predict([r.seq for r in record_list])
            for r,p,c,pr in zip(record_list, pred, classif, proba):
                if (float(p) > 0.5):
                    c = classification(c)
                    if c == "Not determined":
                        out_handle.write(f"{r.id}{sep}{p.item():.5f}{sep}{c}{sep}-{sep}-{sep}-\n")
                    else:
                        out_handle.write(f"{r.id}{sep}{p.item():.5f}{sep}{c}{sep}{pr[0]:.3f}{sep}{pr[1]:.3f}{sep}{pr[2]:.3f}\n")
                else:
                    out_handle.write(f"{r.id}{sep}{p.item():.5f}{sep}-{sep}-{sep}-\n")

def classification(num):
    c = "-"
    if num == 1:
        c = "Membrane"
    elif num == 2:
        c = "Replication"
    elif num == 3:
        c = "Assembly"
    elif num == 0:
        c = "Not determined"
    return c

def gen_feature_for_fasta(fasta):
    fastas = readFasta(str(fasta))
    lambdaValue = 6 
    
    encodings = PAAC(fastas, lambdaValue)
    dpc_enc = DPC(fastas)
    for j in dpc_enc:
        del j[0]

    for i in range(len(encodings)):
                encodings[i].extend(dpc_enc[i])
    encodings[0][0] = "peptide_id"
    return encodings

def predict_AVP(encodings,clf,scaler): 
    peptides = []
    smaller_peptides = []
    for i in range(1,len(encodings)):
        peptide = encodings[i]
        if peptide[1] != 100 :
            peptide.pop(0)
            pep_scaled = scaler.transform([peptide])
            peptides.append(pep_scaled[0])
        else:
            smaller_peptides.append(i)
            peptides.append(peptide)
    pred = clf.predict(peptides)

    for i in smaller_peptides:
        pred[i-1] = 0 
    
    prob = clf.predict_proba(peptides)
    '''
    proba = []
    for i in range(len(prob)):
        proba.append([0,0,0])
        for j in range(len(prob[i])):
            proba[i][j] = prob[i][j] * 100
    '''
    return pred,prob

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

    #########
    encodings = gen_feature_for_fasta(str(args.input_file))
    clas,proba = predict_AVP(encodings,clf,scaler)
    #print(clas)
    ############

    out_handle.write(f"ID{sep}SCORE{sep}CLASS{sep}Membrane{sep}Replication{sep}Assembly\n")
    evaluateSequences(model, parser, out_handle, args.n_batch, sep, clas, proba)

    print("Finished predictions")
    in_handle.close()
    out_handle.close()
