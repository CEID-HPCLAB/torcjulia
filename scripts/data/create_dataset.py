import os
import argparse

import math

from sklearn.model_selection import train_test_split
import pandas as pd

from classification import make_classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", help = "Number of samples", type = int, default = 50000, required = False)  
    parser.add_argument("--features", help = "Number of features", type = int, default = 5, required = False) 
    parser.add_argument("--chunks", help = "chunk size", default = 10, type = int, required = False)
    parser.add_argument("--classes", help = "Number of classes", default = 2, type = int, required = False)
    parser.add_argument("--path", help = "New Data File's path", default = "./", type = str, required = False)
    parser.add_argument("--problem_name", "-pbn", help = "problem name", default = "xyz_40K", type = str, required = False) 

    args = parser.parse_args()
    
    n_samples = args.samples; n_features = args.features 

    test_size = 0.2
    
    n_chunks = args.chunks; path = args.path

    base_train_size, remainder = divmod(n_samples, n_chunks)
    
    dataset_name = args.problem_name; n_classes = args.classes

    n_samples = math.ceil(n_samples / (1 - test_size))
    
    train_file = os.path.join(path, f'{dataset_name}.csv')

    if os.path.exists(train_file):
        os.remove(train_file)

    use_index = False; reset_index = False 

    # chunk_size = n_samples // n_chunks  

    for i in range(n_chunks):
        chunk_train_size = base_train_size + (1 if i < remainder else 0)

        chunk_total = math.ceil(chunk_train_size / (1 - test_size))

        X, y = make_classification(n_samples = chunk_total, n_features = n_features, n_informative = n_features, n_redundant = 0,
                                   n_clusters_per_class = 1, flip_y = 0.01, n_classes = n_classes, random_state = 42+i,
                                   shuffle = True) 
        y = y.reshape(-1, 1)
        
        df1 = pd.DataFrame(data = X)
        df2 = pd.DataFrame(data = y)

        df = pd.concat([df1, df2], axis = 1)

        if i == 0:
            use_header = True
        else:
            use_header = False
        
        df.columns =[f"x{i}" for i in range(n_features)] + [f"y{i}" for i in range(y.shape[1])]

        train, test = train_test_split(df, train_size = chunk_train_size)
        train.to_csv(train_file, mode = "a", header = use_header, index = use_index, float_format = "%.6f")