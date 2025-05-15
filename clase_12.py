import wooldridge as wd
import pickle
import sklearn
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeRegressor
wase = wd.data("gpa1")
wase.info()
gpa = wase[["colGPA","hsGPA","ACT","age"]]

vardep = gpa["colGPA"]
varinds = gpa[["hsGPA","ACT","age"]]

dtreg = DecisionTreeRegressor(max_depth=3)
dtreg.fit(varinds, vardep)

with open("model.pickle", "wb") as f:
    pickle.dump(dtreg, f)


