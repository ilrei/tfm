import streamlit as st
from functions import(
    add_thresh_slider,
    load_data,
    get_metrics,
    get_medium_metrics,
    lineplot_medium_metrics,
    barplot_medium_metrics
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from sklearn.metrics import *

st.set_page_config(page_title="Lead Scorer", layout="wide")

######create sidebar
threshold = add_thresh_slider()

######load_data
df_train, df_test = load_data(threshold)

######get metrics
train_accuracy, train_f1, train_tpr, train_tnr, \
test_accuracy, test_f1, test_tpr, test_tnr = get_metrics(df_train, df_test)

df_acc, df_f1, df_tpr, df_tnr = get_medium_metrics(df_train, df_test)

######title and intro
st.title('Lead Scorer classifier model')

with st.container():
    st.subheader(
        """
        Here you can see how the model predicts leads to become bookings!
        """)
    st.write(
        """
        <div style=background-color:Gainsboro;width:100%;max-width:820px;height:120%;min-height:130px;padding-left:3%;><br>
        📈 Check model performance through accuracy and f1 scores, average and by traffic source.<br>
        🧪 See how the model performs on known (train) and unseen (test) data.<br>
        🖱️ Scroll down to check positives and negatives predictions accuracy by traffic source.</div><br>
        """,
          unsafe_allow_html=True,
    )


######f1 & accuracy
st.subheader('How well the model predicts?')

with st.container():
    col1, col2, col3 = st.columns(3)
    col1.subheader("Accuracy score")
    col2.subheader("F1 Score")

with st.container():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy on Test", "{:.2%}".format(test_accuracy), "{:.3%}"\
            .format((test_accuracy - train_accuracy)/test_accuracy))
    col2.metric("Accuracy on Train", "{:.2%}".format(train_accuracy))
    col3.metric("F1 Score on Test", "{:.2%}".format(test_f1), "{:.3%}"\
                .format((test_f1 - train_f1)/test_f1))
    col4.metric("F1 Score on Train", "{:.2%}".format(train_f1))

with st.container():
    col1, col2, col3 = st.columns(3)
    col1.pyplot(lineplot_medium_metrics(df_acc))
    col2.pyplot(lineplot_medium_metrics(df_f1))
    col3.write(
        """
        <div style=background-color:PowderBlue;width:100%;max-width:520px;height:100%;min-height:200px;padding-top:7%;padding-left:5%;padding-right:3%;>
        <b>Accuracy score:</b> How  many predictions were actually correct? </br></br>
        <b>F1 score:</b> Weighted average of Precision (correct positive predictions over total positive predicted)
         and Recall (correct positive predictions over total real positive)
        """,
          unsafe_allow_html=True,
    )


######tpr & fpr
st.subheader('How well it detects positives & negatives?')

with st.container():
    col1, col2, col3 = st.columns(3)
    col1.subheader("True Positive rate")
    col2.subheader("True Negative rate")

with st.container():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("TPR on Test", "{:.2%}".format(test_tpr), "{:.3%}"\
            .format((test_tpr - train_tpr)/test_tpr))
    col2.metric("TPR on Train", "{:.2%}".format(train_tpr))
    col3.metric("TNR on Test", "{:.2%}".format(test_tnr), "{:.3%}"\
                .format((test_tnr - train_tnr)/test_tnr))
    col4.metric("TNR on Train", "{:.2%}".format(train_tnr))

with st.container():
    col1, col2, col3 = st.columns(3)
    col1.pyplot(barplot_medium_metrics(df_tpr.sort_values(by='tpr_test', ascending=False)))
    col2.pyplot(barplot_medium_metrics(df_tnr.sort_values(by='tnr_test', ascending=False)))
    col3.write(
        """
        <div style=background-color:PowderBlue;width:100%;max-width:520px;height:100%;min-height:200px;padding-top:10%;padding-left:5%;padding-right:3%;>
        <b>True Positive Rate:</b> How many positives is the model able to predict?</br></br>
        <b>True Negative Rate:</b>  How many negatives is the model able to predict?
        """,
          unsafe_allow_html=True,
    )