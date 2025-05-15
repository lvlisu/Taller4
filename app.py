import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import numpy as np

st.title("GPA de la universidad según el GPA de la ecundaria, la edad del estudiante y el puntaje de ACT")
tab1, tab2, tab3 = st.tabs(["Análisis Univariado","Análisis Bivariado", "Análisis interactivo"])
datos = pd.read_csv("wooldridge.csv")

with open("model.pickle", "rb") as f:
    modelo = pickle.load(f)
with tab1:
    fig, ax = plt.subplots(1,4, figsize = (10,4))
    ax[0].hist(datos["colGPA"])
    ax[0].set_title("GPA universidad")
    ax[1].hist(datos["hsGPA"])
    ax[1].set_title("GPA secundaria")
    ax[2].hist(datos["ACT"])
    ax[2].set_title("Puntaje del ACT")
    ax[3].hist(datos["age"])
    ax[3].set_title("Edad")
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(1,3, figsize = (10,4))
    sns.scatterplot(x = "ACT", y = "colGPA", data=datos, ax=ax[0])
    ax[0].set_xlabel("Puntaje ACT")
    ax[0].set_ylabel("GPA universidad")
    ax[0].set_title("SALARIO - GENERO")
    sns.scatterplot(x = "hsGPA", y = "colGPA", data=datos, ax=ax[1])
    ax[1].set_xlabel("GPA secundaria")
    ax[1].set_ylabel("GPA universidad")
    ax[1].set_title("GPA universidad - GPA secundaria")
    sns.scatterplot(x = "age", y = "colGPA", data=datos, ax=ax[2])
    ax[2].set_xlabel("Edad")
    ax[2].set_ylabel("GPA universidad")
    ax[2].set_title("Edad - GPA universidad")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    hsgpa = st.slider("HSGPA", 0.0, 4.0)
    age = st.slider("EDAD", 15, 100)
    act = st.slider("Puntaje ACT", 1, 36)
    if st.button("Predecir"):
        pred = modelo.predict(np.array([[hsgpa, age, act]]))
        st.write(f"Su promedio sería {round(pred[0], 1)}")
