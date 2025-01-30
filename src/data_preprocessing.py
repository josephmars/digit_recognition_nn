import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def load_and_preprocess_data(filepath, random_seed=0):
    # Load data into a list
    data = pd.read_csv(filepath)
    data = data.values.tolist()
    
    # Remove first 19 columns from data and store in metadata
    metadata = data[:19]
    data = data[19:]
    
    # Iterate through data, append 32 rows to each data point and the 33rd row to the label
    x = ''
    X = []
    y = []
    for i in range(len(data)):
        row = data[i][0].strip()
        
        if len(row) == 32:
            x += row
        elif len(row) == 1:
            y.append(int(row))
            X.append(x)
            x = ''
    
    # Split each string into a list of characters
    X = [list(map(int, x)) for x in X]
            
    # Split data into training, testing, and validation sets (70-15-15)        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_seed)
    
    # Convert to NumPy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)
    
    # One-hot encoding
    y_train_one_hot = one_hot_encode(y_train)
    y_test_one_hot = one_hot_encode(y_test)
    y_val_one_hot = one_hot_encode(y_val)
    
    return X_train, X_test, X_val, y_train, y_test, y_val, y_train_one_hot, y_test_one_hot, y_val_one_hot

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.flatten()] = 1
    return one_hot

def apply_pca(X_train, X_test, X_val, num_components=100):
    pca = PCA(n_components=num_components, random_state=0)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_test_pca, X_val_pca 