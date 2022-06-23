import time
import os
import pathlib

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
from torch.optim import RMSprop

import joblib
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor

from .util import getBatch, timeSince, IUPAC
from .features import calcPCP, calcPseAAC


class LSTM2D(nn.Module):
    classes = ["POSITIVE", "NEGATIVE"]

    def __init__(self,
                 embedding_size: int = 128,
                 hidden_layer_size: int = 100,
                 output_size: int = 1,
                 num_layers: int = 5,
                 vocab: list = IUPAC,
                 dropout: float = 0.0,
                 threshold: float = 0.5,
                 dev='cpu'):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab = vocab
        self.threshold = threshold

        self.dev = torch.device(dev)

        self.embedding = nn.Embedding(len(vocab), embedding_size)

        self.lstm = nn.LSTM(embedding_size,
                            hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True)

        self.linear = nn.Linear(2*hidden_layer_size, output_size)

        self.sigmoid = nn.Sigmoid()

        self.to(self.dev)

    def forward(self, input_batch:list):

        # input_batch is a list of tensors with indices of aa's
        # Here we convert this indices to embeddings at aa level
        seq_embeddings = [
            self.embedding(seq.to(self.dev))
            for seq in input_batch
        ]

        # Convert to a PackedSequence object
        # (allows variable sequence lengths)
        packed_input = pack_sequence(
            seq_embeddings,
            enforce_sorted = False
        )

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_input)
        
        # Convert back from PackedSequence to tensors
        # (0 padded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        lengths = torch.tensor([seq.shape[0] for seq in input_batch])

        # Break both directions of lstm and select "last" hidden output
        # (in the reverse direction it's the first output)

        # Deal with padding
        out_forward = torch.vstack([
            output[i,l-1,:self.hidden_layer_size]
            for i,l in enumerate(lengths)
        ])

        out_reverse = output[:, 0,   self.hidden_layer_size:]
        
        # Concatenate again
        output = torch.cat((out_forward, out_reverse), 1)

        # Convert to single score for each sequence
        output = self.linear(output)
        output = self.sigmoid(output)        
        return output

    def trainDataset(
        self,
        X: torch.tensor,
        y: torch.tensor,       
        learning_rate: float=0.05,
        batch_size: int = 8,
        epochs: int = 1,
        show_results: bool = False,
        print_every: int = 2,
        criterion: nn.modules.loss._Loss = nn.BCELoss(),
        optimizer_f: torch.optim.Optimizer = torch.optim.SGD):
        '''Trains the Network on _data_ using _learning_rate_. Other params:
        
            epochs : number of epochs of training. default = 1.
            criterion : loss function to be utilized. default = nn.NLLLoss()
            optimizer_f : optimizer function to be constructed with the network parameters. default = SGD
            
        Printing results: Will print results based on _show_results_ value. Will 
        print info about the first example of the batch, every _print_every_ batches.
        '''

        start = time.time()
        optimizer = optimizer_f(self.parameters(), lr=learning_rate)

        size = len(X)
        for ep in range(1, epochs+1):
            total_loss = 0

            if show_results:
                print('Started epoch [{:02}]'.format(ep))
                i = 0

            #permutation of all indexes
            permutation = torch.randperm(size)

            for batch_i in range(0,size,batch_size):


                optimizer.zero_grad()
                
                b_x = getBatch(X, permutation[batch_i:batch_i+batch_size])
                b_y = getBatch(y, permutation[batch_i:batch_i+batch_size])
                
                b_y = torch.vstack(b_y).float()

                outputs = self.forward(b_x)

                loss = criterion(outputs,b_y.to(self.dev))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                if show_results:
                    if i % print_every == 0:
                        self._logExample(
                            guess = self._output2class(outputs[0],self.threshold),
                            category = self._label2class(b_y[0]),
                            seq=self._indxtensor2str(b_x[0]),
                            size=size,
                            s_time = start,
                            loss=loss,
                            i=i*batch_size)
                        
                    i += 1
        return total_loss

    def _output2class(self, tens:torch.tensor, threshold:float) -> str:
        return LSTM2D.classes[0 if tens < threshold else 1]

    def _label2class(self, tens:torch.tensor) -> str:
        return LSTM2D.classes[tens.int()]

    def _indxtensor2str(self, tens:torch.tensor) -> str:
        return ''.join(self.vocab[i] for i in tens)
   
    def _str2indxtensor(self, seq:str) -> torch.tensor:
        return torch.tensor([self.vocab.index(s) for s in seq])
   
    def _logExample(
        self,
        guess: str,
        category:str,
        seq:str, 
        size: int,
        s_time,
        loss:torch.tensor,
        i:int):

        correct = "✓" if guess == category else "✗ ({})".format(
            category)

        print("{} {:.2f} % ({}) {:.04f} {} / {} {}".format(
            i, i / size * 100, timeSince(s_time), loss, seq, guess, correct))

    def predict(self, seq_list: list):
        idx_list = [self._str2indxtensor(seq) for seq in seq_list]

        return self(idx_list)


class MLmodel():
    '''
    Wrapper to RandomForest regressor.
    '''
    def __init__(self, 
                 n_estimators:int = 100, 
                 max_depth:int = 2,
                 min_samples_split:int = 6,
                 max_features:str = 'auto'):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features


        self.model = RandomForestRegressor(
            n_estimators=n_estimators,max_features=max_features,
            max_depth=max_depth,min_samples_split=min_samples_split)

    def trainDataset(self, X: np.array, y: np.array):
        self.model.fit(X,y)

    def predict(self, X: np.array):
        return self.model.predict(X)


class EnsembleModel():


    def get_feature_function(self, fset):
        _feature_f_dict = {
            "PCP": (lambda seq : calcPCP(seq.replace("X",""))),
            "PseAAC": (lambda seq : calcPseAAC(seq.replace("X",""))),
            "PseAAC+PCP": (
                lambda seq : np.hstack(
                    (calcPCP(seq.replace("X","")),
                    calcPseAAC(
                        seq.replace("X",""), 
                        self.pseaac_lambda, 
                        self.pseaac_float))
                ))
        }
        return _feature_f_dict[fset]

    def __init__(
        self, 
        weight:float = 0.7,
        embedding_size: int = 512,
        hidden_layer_size: int = 128,
        output_size: int = 1,
        num_layers: int = 7,
        vocab: list = IUPAC,
        dropout: float = 0.109917,
        dev: str='cpu',
        n_estimators: int = 100, 
        max_depth: int = 2,
        min_samples_split: int = 6,
        max_features: str = 'auto',
        feature_set="PCP",
        pseaac_lambda=None,
        pseaac_float=None,
    ):
        
        self.hidden_layer_size = hidden_layer_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dev = dev
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.weight = weight
        self.vocab = vocab
        self.feature_set = feature_set
        self.pseaac_lambda = pseaac_lambda
        self.pseaac_float = pseaac_float

        self.feature_function = self.get_feature_function(self.feature_set)

        self.dl_model = LSTM2D(
            hidden_layer_size=hidden_layer_size,
            embedding_size=embedding_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            vocab=vocab,
            dev=dev,
            )
        
        self.ml_model = MLmodel(
            max_depth=max_depth,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split)

    def trainDataset(
        self,
        pos_seqs: list,
        neg_seqs: list,
        n_epochs: int,
        lr: float, 
        optimizer: torch.optim.Optimizer = RMSprop,
        fit: bool = True,
        batch_size : int = 8,
        show_results: bool = False):

        X_t = [self.dl_model._str2indxtensor(seq) for seq in pos_seqs] +\
            [self.dl_model._str2indxtensor(seq) for seq in neg_seqs]

        y_t = torch.hstack([torch.ones(len(pos_seqs)),torch.zeros(len(neg_seqs))])

        loss = self.dl_model.trainDataset(
            X=X_t, y=y_t, learning_rate=lr,
            epochs=n_epochs, optimizer_f=optimizer,
            show_results=show_results, batch_size=batch_size
        )
        if fit:
            X = np.vstack(
                [self.feature_function(seq) for seq in pos_seqs] + 
                [self.feature_function(seq) for seq in neg_seqs])
            y = np.hstack([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))])
            self.ml_model.trainDataset(X, y)

        return loss

    def predict(self, sequence_list: list) -> torch.tensor:
        feature_rep = np.vstack([
            self.feature_function(seq) for seq in sequence_list
        ])

        ml_predictions = self.ml_model.predict(feature_rep)
        dl_predictions = self.dl_model.predict(sequence_list)

        # combine predictions

        final_predictions = [
            (dl_predictions[i]*self.weight +
            ml_predictions[i]*(1-self.weight))
            for i in range(len(sequence_list))
        ]

        return torch.tensor(final_predictions)

    def save(self, dirr: str):
        """Saves trained model in _dirr_."""
        pathlib.Path(dirr).mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.dl_model.state_dict(),
            "params": {
                "weight": self.weight,
                "embedding_size": self.embedding_size,
                "hidden_layer_size": self.hidden_layer_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "n_estimators" : self.n_estimators,
                "max_depth" : self.max_depth,
                "max_features" : self.max_features,
                "min_samples_split" : self.min_samples_split,
                "feature_set": self.feature_set,
                "pseaac_lambda": self.pseaac_lambda,
                "pseaac_float": self.pseaac_float,
            }
        }
        torch.save(state, os.path.join(dirr,"lstm_module.pt"))
        joblib.dump(self.ml_model, os.path.join(dirr,"rf_module.jlib"))

    @staticmethod
    def load_from(dirr: str, dev: str = "cpu"):
        state = torch.load(
            os.path.join(dirr,"lstm_module.pt"),
            map_location=torch.device(dev))

        p = state["params"]
        model = EnsembleModel(
                weight = p["weight"],
                embedding_size = p["embedding_size"],
                hidden_layer_size = p["hidden_layer_size"],
                num_layers = p["num_layers"],
                dropout = p["dropout"],
                n_estimators = p["n_estimators"],
                max_depth = p["max_depth"],
                max_features = p["max_features"],
                min_samples_split = p["min_samples_split"],
                feature_set=p.get("feature_set") if p.get("feature_set") else "PCP", 
                pseaac_lambda=p.get("pseaac_lambda"),
                pseaac_float=p.get("pseaac_float"),
                dev = dev,
        )
        model.dl_model.load_state_dict(state["model_state_dict"])

        model.ml_model = joblib.load(os.path.join(dirr,"rf_module.jlib"))

        return model