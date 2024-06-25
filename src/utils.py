import pandas as pd
import torch


def load_data(proportion_validation = 0.2):
    df = pd.read_csv("./data/creditcard.csv")
    df["Class"] = df["Class"].astype("int32")
    df.drop(columns="Time", inplace=True)

    mask = df["Class"] == 1
    df_fraud = df[mask]
    df = df[~mask]
    y = torch.tensor(df["Class"].values)
    df.drop(columns="Class", inplace=True)
    X = torch.tensor(df.values)

    mask = torch.rand((X.size(0), )) >= proportion_validation
    X_train = X[mask,:]
    y_train = y[mask]
    X_val = X[~mask, :]
    y_val = y[~mask]

    y_val_fraud = torch.tensor(df_fraud["Class"].values)
    df_fraud.drop(columns="Class", inplace=True)
    X_val_fraud = torch.tensor(df_fraud.values)

    y_val = torch.cat((y_val, y_val_fraud), dim=0)
    X_val = torch.cat((X_val, X_val_fraud), dim=0)

    return (X_train.float(), y_train.long()), (X_val.float(), y_val.long())

def true_positive(y, y_hat, label):
    return torch.logical_and(y == label, y_hat == label).sum()

def false_positive(y, y_hat, label):
    return torch.logical_and(y != label, y_hat == label).sum()

def false_negative(y, y_hat, label):
    return torch.logical_and(y == label, y_hat != label).sum()

def f1_score(true_positive, false_positive, false_negative):
    if true_positive == 0:
        return 0
    true_positive = 2*true_positive
    return true_positive / (true_positive + false_positive + false_negative)