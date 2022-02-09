import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from sklearn.metrics import *

def add_thresh_slider():
    threshold = st.sidebar.slider('Select threshold', min_value=0.05, max_value=0.95, value=0.5, step=0.05)
    return (threshold)

@st.cache
def load_data(threshold):
    train_csv = 'data_files/scorer_train_predicts.csv'
    train_predicts = pd.read_csv(train_csv)
    train_predicts['ypred_thresh'] = [1 if i >= threshold else 0 for i in train_predicts['y_pred_prob']]

    test_csv = 'data_files/scorer_test_predicts.csv'
    test_predicts = pd.read_csv(test_csv)
    test_predicts['ypred_thresh'] = [1 if i >= threshold else 0 for i in test_predicts['y_pred_prob']]

    return(train_predicts, test_predicts)


def get_metrics(df_train, df_test):
    ytrain = df_train['is_booking']
    ypred_train = df_train['ypred_thresh']

    ytest = df_test['is_booking']
    ypred_test = df_test['ypred_thresh']

    train_accuracy = accuracy_score(ytrain, ypred_train)
    train_f1 = f1_score(ytrain, ypred_train)
    train_tpr = recall_score(ytrain, ypred_train)
    train_tnr = recall_score(ytrain, ypred_train, pos_label=0)

    test_accuracy = accuracy_score(ytest, ypred_test)
    test_f1 = f1_score(ytest, ypred_test)
    test_tpr = recall_score(ytest, ypred_test)
    test_tnr = recall_score(ytest, ypred_test, pos_label=0)

    return (train_accuracy, train_f1, train_tpr, train_tnr, test_accuracy, test_f1, test_tpr, test_tnr)


def get_medium_metrics(df_train, df_test):
    mediums = df_test['clean_medium'].value_counts().index.tolist()

    f1s_train = []
    accuracies_train = []
    tprs_train = []
    tnrs_train = []

    f1s_test = []
    accuracies_test = []
    tprs_test = []
    tnrs_test = []

    for medium in mediums:
        ytrain = df_train.loc[df_train['clean_medium'] == medium]['is_booking']
        ypred_train = df_train.loc[df_train['clean_medium'] == medium]['ypred_thresh']

        ytest = df_test.loc[df_test['clean_medium'] == medium]['is_booking']
        ypred_test = df_test.loc[df_test['clean_medium'] == medium]['ypred_thresh']

        # calculate and append acc on train
        acc_train = accuracy_score(ytrain, ypred_train)
        accuracies_train.append(acc_train)
        # calculate and append f1 on train
        f1_train = f1_score(ytrain, ypred_train)
        f1s_train.append(f1_train)
        # calculate and append tpr on train
        tpr_train = recall_score(ytrain, ypred_train)
        tprs_train.append(tpr_train)
        # calculate and append tnr on train
        tnr_train = recall_score(ytrain, ypred_train, pos_label=0)
        tnrs_train.append(tnr_train)

        # calculate and append acc on test
        acc_test = accuracy_score(ytest, ypred_test)
        accuracies_test.append(acc_test)
        # calculate and append f1 on test
        f1_test = f1_score(ytest, ypred_test)
        f1s_test.append(f1_test)
        # calculate and append tpr on test
        tpr_test = recall_score(ytest, ypred_test)
        tprs_test.append(tpr_test)
        # calculate and append tnr on test
        tnr_test = recall_score(ytest, ypred_test, pos_label=0)
        tnrs_test.append(tnr_test)

    df_acc = pd.DataFrame(np.column_stack([accuracies_train, accuracies_test]),
                          columns=['accuracy_train', 'accuracy_test'], index=mediums)
    df_f1 = pd.DataFrame(np.column_stack([f1s_train, f1s_test]), columns=['f1_train', 'f1_test'], index=mediums)
    df_tpr = pd.DataFrame(np.column_stack([tprs_train, tprs_test]),
                          columns=['tpr_train', 'tpr_test'], index=mediums)
    df_tnr = pd.DataFrame(np.column_stack([tnrs_train, tnrs_test]), columns=['tnr_train', 'tnr_test'], index=mediums)

    return (df_acc, df_f1, df_tpr, df_tnr)


def lineplot_medium_metrics(df):
    fig, ax = plt.subplots(figsize=(5, 2))
    plt.plot(df.index, df[df.columns[0]], label='train set')
    plt.plot(df.index, df[df.columns[1]], label='test set',
             color='red',
             linewidth=1.0,
             linestyle='--'
             )
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='center right', fontsize='small')
    return(fig)


def barplot_medium_metrics(df):
    fig, ax = plt.subplots(figsize=(5, 2))
    plt.bar(df.index, df[df.columns[0]], label='train set')
    plt.bar(df.index, df[df.columns[1]], label='test set', color='red')
    ax.set_xticklabels(df.index, rotation=45, horizontalalignment='right')
    ax.set_ylim(0,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='center right', fontsize='small')
    return(fig)