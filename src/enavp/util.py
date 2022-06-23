import time
import math
from Bio.Data.IUPACData import protein_letters

AVP_PARAMS = {
    "fset": "PCP",
    "weight": 0.7,
    "n_layers": 7,
    "embedding_size": 512,
    "hidden_layer_size": 128,
    "dropout": 0.109917,
    "n_estimators": 50,
    "max_depth": 2,
    "optimizer": "RMSprop",
    "lr": 0.000486018,
}

AMP_PARAMS = {
    "fset": "PCP",
    "weight": 0.7,
    "n_layers": 6,
    "embedding_size": 256,
    "hidden_layer_size": 64,
    "dropout": 0.127588,
    "n_estimators": 96,
    "max_depth": 2,
    "optimizer": "RMSprop",
    "lr": 0.00113432,
}

IUPAC = protein_letters + "X"

def getBatch(data, indexes):
    '''util to extract batch of examples from data, 
    returns values,labels
    '''
    return [data[i] for i in indexes]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s}s"


def partialDictRoundedSum(d, keys):
    return round(sum([d[k] for k in keys]), 3)


def nextn(it, n, truncate=False):
    """Util to get n elements from iterator at a time"""
    assert n > 0
    cur_batch = []
    while True:
        try:
            cur_batch.append(next(it))
        except StopIteration:
            break
        if len(cur_batch) == n:
            yield cur_batch
            cur_batch = []

    if len(cur_batch) and not truncate:
        yield cur_batch


def validateSeqs(record_list, unknown_id_count, warned):
    for record in record_list:
        other_aas = [aa for aa in record.seq if aa not in IUPAC]
        if len(other_aas) > 0:
            if not warned:
                warnings.warn("Non standard AAs will be replaced by X "
                            "for predictions.")
                warned = True
            for aa in set(other_aas):
                record.seq = record.seq.replace(aa,"X")

        if record.id == "":
            record.id = f"unknown_seq_{unknown_id_count}"
            unknown_id_count += 1
    return record_list, unknown_id_count, warned
