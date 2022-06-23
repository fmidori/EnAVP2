import numpy as np
from .util import partialDictRoundedSum
from Bio.SeqUtils.ProtParamData import hw, ta, kd
from Bio.Data.IUPACData import protein_letters, protein_weights
import re, sys, os, platform
import math
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from .iFeature import readFasta,checkFasta,minSequenceLength,minSequenceLengthWithNormalAA

# Normalized Hidrophobicity
# Original Values from Tanfordl J. Am. Chem. Soc. 84:4240-4274(1962)
H_1_sum = sum(ta.values())
H_1_std = sum([(ta[aa] - H_1_sum) / 20 for aa in protein_letters])
H_1 = {aa: (ta[aa] - H_1_sum) / H_1_std for aa in protein_letters}

# Normalized Hidrophilicity
# Original Values from Hopp and Wood, Proc. Natl. Acad. Sci. U.S.A. 78:3824-3828(1981)
H_2_sum = sum(hw.values())
H_2_std = sum([(hw[aa] - H_2_sum) / 20 for aa in protein_letters])
H_2 = {aa: (hw[aa] - H_2_sum) / H_2_std for aa in protein_letters}

# Normalized side chain weights
M_sum = sum(protein_weights.values())
M_std = sum([(protein_weights[aa] - M_sum) / 20 for aa in protein_letters])
M = {aa: (protein_weights[aa] - M_sum) / M_std for aa in protein_letters}


def _calc_correlation(aa1: str, aa2: str, scales: list) -> float:
        return sum([(s[aa1] - s[aa2]) ** 2 for s in (H_1, H_2, M)]) / 3


def get_aac(seq: str) -> dict:
    l = len(seq)
    d = {aa: seq.count(aa)/l for aa in protein_letters}
    return d


def calcPseAAC(seq: str, l_param: int, weight: float) -> np.array:
    L = len(seq)
    if L == 0:
        return np.zeros(20+l_param)

    theta = np.zeros(l_param)
    for j in range(1, l_param + 1):
        t_j = sum(
            [
                _calc_correlation(seq[i], seq[i + j])
                for i in range(L - j)
            ]
        )
        theta[j - 1] = t_j / L - j

    # Sum of all AAC values is one by definition
    sum_term = 1 + weight * sum(theta)

    aac = get_aac(seq)
    # First 20 terms reflect the effect of AAC
    X = [aac[k] / sum_term for k in protein_letters]

    # Components 21 to 20 + lambda reflect the effect of sequence order
    X += [weight * t / sum_term for t in theta]

    return np.array(X)


def calcPCP(seq: str) -> np.array:
    seq_size = len(seq)
    if seq_size == 0:
        return np.zeros(35)

    net_charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
                  'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
                  'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
                  'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}

    net_hydrogen = {'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
                    'G': 0, 'H': 1, 'I': 0, 'K': 2, 'L': 0,
                    'M': 0, 'N': 2, 'P': 0, 'Q': 2, 'R': 4,
                    'S': 1, 'T': 1, 'V': 0, 'W': 1, 'Y': 1}

    aa_counts = {k: seq.count(k) for k in protein_letters}

    aa_perc = get_aac(seq)

    # PROPERTIES Q-P

    aliphatic = partialDictRoundedSum(aa_perc, 'IVL')

    negative_charged = partialDictRoundedSum(aa_perc, 'DE')

    total_charged = partialDictRoundedSum(aa_perc, 'DEKHR')

    aromatic = partialDictRoundedSum(aa_perc, 'FHWY')

    polar = partialDictRoundedSum(aa_perc, 'DERKQN')

    neutral = partialDictRoundedSum(aa_perc, 'AGHPSTY')

    hydrophobic = partialDictRoundedSum(aa_perc, 'CFILMVW')

    positive_charged = partialDictRoundedSum(aa_perc, 'KRH')

    tiny = partialDictRoundedSum(aa_perc, 'ACDGST')

    small = partialDictRoundedSum(aa_perc, 'EHILKMNPQV')

    large = partialDictRoundedSum(aa_perc, 'FRWY')

    # SCALES

    kyleD = round(
        sum(
            [aa_counts[k]*kd[k] for k in protein_letters]
        )/seq_size, 3)

    molW = round(
        sum([aa_counts[k]*protein_weights[k] for k in protein_letters]), 3)

    netCharge = sum([aa_counts[k]*net_charge[k] for k in protein_letters])

    netH = round(
        sum([aa_counts[k]*net_hydrogen[k] for k in protein_letters]), 3)

    results =  [netH, netCharge, molW, kyleD] 
    results += [v for v in aa_perc.values()]
    results += [
        tiny, small, large, aliphatic, aromatic,
        total_charged, negative_charged, positive_charged,
        polar, neutral, hydrophobic
    ]

    return np.array(results)

def DPC(fastas): #**kw
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	#AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		code = code + tmpCode
		encodings.append(code)
	return encodings

def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=6, w=0.5, **kw): #changed here before: lambda=30 and w=0.05 
	seqlen = minSequenceLengthWithNormalAA(fastas)
	if seqlen < lambdaValue + 1:
		print('Warning: Some sequences are smaller than the lambdaValue: ' + str(lambdaValue + 1) + '\n\n')
		#return 0
		#lambdaValue = seqlen - 1
	#dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/PAAC.txt'
	dataFile='src/enavp/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])

	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		if len(sequence) >= 7:
			code = [name]
			theta = []
			for n in range(1, lambdaValue + 1):
				theta.append(
					sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
					len(sequence) - n))
			myDict = {}
			for aa in AA:
				myDict[aa] = sequence.count(aa)
			code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
			code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
			encodings.append(code)
		else:
			#print(name)
			encodings.append([100] * 26)
		#print(len(encodings[0]))
	return encodings