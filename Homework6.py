import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

st.title("KMeans Clustering with Random Blobs")
st.markdown("[Alfonso Vieyra](https://github.com/anvieyra)")
X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)

if 'df' in st.session_state:
    df = st.session_state['df']
else:     
    df = pd.DataFrame(X, columns = list("ab"))
    st.session_state['df'] = df
    
starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])
slider = st.slider(label = "Select Max iterations", min_value = 1)
kmeans = KMeans(n_clusters = 5, max_iter=slider, init=starting_points, n_init = 1)
kmeans.fit(X);
df["c"] = kmeans.predict(X)


chart1 = alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "c:N"
)

df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))

chart_centers = alt.Chart(df_centers).mark_point().encode(
    x = "a",
    y = "b",
    color = alt.value("black"),
    shape = alt.value("diamond"),
    )

chart1 + chart_centers