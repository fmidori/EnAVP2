import subprocess
import numpy as np
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def gen_features(feature): 
    subprocess.run([
                    "python3", 
                    "../iFeature/iFeature.py",
                    "--file", "avps_labeledV3.cdhit.pep",
                    "--type" , str(feature),
                    "--out", str(feature) + ".tsv" ] ,
                    stdout=subprocess.DEVNULL)
    return str(feature) + ".tsv" 

def create_X_and_y(tsv,num_of_labels):
    labels = [i+1 for i in range(int(num_of_labels))]
    with open(tsv, 'r') as t:
        X = []
        y = []
        for line in t:
            if not line.startswith('#'):
                label = int(line.split('\t')[0].split('|')[1])
                if label in labels:
                    y.append(label)
                    x = line.replace('\t',',').replace('\n','').split(',')
                    x.pop(0)
                    x = [float(i) for i in x]
                    X.append(x)
        y = np.array(y)
    return X,y

def merge_features(feature,num_of_labels):
    X = []
    y = []
    #for feature in ["PAAC+PCP"]:
        #if "+" in feature:
    feature_list = feature.split("+")
    for feat in feature_list:
                #print(feat)
        X_to_merge,y = create_X_and_y(feat + ".tsv",num_of_labels)
        if not X:
            X = X_to_merge
        else:
            for i in range(len(X)):
                X[i].extend(X_to_merge[i])
            #print(len(X[0]))
    return X,y 

def apply_pca(X):
    pca = PCA(n_components = 0.95, random_state=29)
    X_pca = pca.fit_transform(X)
    print("number of components X:", len(X_pca[0]))
    #print("number of components X_test:", len(X_test_pca[0]))
    return X_pca

def smote(X_train,y_train,strategy,kn):
    X_train_smote, y_train_smote = SMOTE(sampling_strategy=strategy, k_neighbors=int(kn),random_state=42).fit_resample(X_train, y_train)
    print("X_train_SMOTE:", len(X_train_smote))
    print("y_train_SMOTE label counts:",np.unique(y_train_smote,return_counts=True))
    return X_train_smote,y_train_smote

def apply_scaler(X):
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess(X,y,do_scaler,do_pca):
    if do_scaler == "yes":
        X = apply_scaler(X)
        print("X scaled")
    
    if do_pca == "yes":
        X = apply_pca(X)
        #print("PCA applied:" , len(X[0]))

    # Train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # apply SMOTE
    X_train,y_train = smote(X_train,y_train,"auto",5)
    return X_train,X_test,y_train,y_test

def preprocess_for_model(X,y,do_scaler):
    # Train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # apply SMOTE
    X_train,y_train = smote(X_train,y_train,"auto",5)
    
    if do_scaler == "yes":
        X_train = apply_scaler(X_train)
        X_test = apply_scaler(X_test)
        print("X scaled")
    
    return X_train,X_test,y_train,y_test


if __name__ == "__main__":
    X,y = merge_features("PAAC+AAC",3)
    #print(len(X[300]))
    X_train,X_test,y_train,y_test = preprocess(X,y,"no","yes")
    print(len(X_train))
