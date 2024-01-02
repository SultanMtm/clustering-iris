import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('IRIS.csv')

x = df.drop(['species'], axis = 1)

st.header("Iris Flower")
st.subheader("Isi dataset")
st.write(x)

#Menampilan elbow
clusters = []
for i in range(1, 10):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Jumlah K")
clust = st.sidebar.slider("pilih jumlah cluster :", 2,6,3,1)

def k_means(n_clust) : 
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='sepal_length', y='petal_length', hue='Labels', size='Labels', markers=True, palette=sns.color_palette('hls', n_colors=n_clust), data=x)

    for label in x['Labels']:
        plt.annotate(label,
                 (x[x['Labels'] == label]['sepal_length'].mean(),
                  x[x['Labels'] == label]['petal_length'].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')
    st.subheader('Cluster Plot')
    st.pyplot()
    st.write(x)
k_means(clust)