import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#creación del DataFrame desde la tabla q hicimos con los datos de nuestros compañeros
data = [
    ['Majo', "Python", 5],
    ['Majo', "Java", 5],
    ['Ezequiel', "Java", 4],
    ['Ezequiel', "Assembler", 4],
    ['Majo', "Python", 3],
    ['Majo', "Java", 3],
    ['Majo', "SQL", 3],
    ['Majo', "Assembler", 1],
    ['Ezequiel', "Python", 1],
    ['Leandro Gimenez', "Java", 3],
    ['Leandro Gimenez', "SQL", 3],
    ['Fernando', "Java", 3],
    ['Pablo Da Silva', "Java", 3],
    ['Pablo Da Silva', "SQL", 3],
    ['Juan', "Java", 3],
    ['Santiago Gonzalez', "Python", 1],
    ['Matias Felipe Bianciotto', "Python", 2],
    ['Matias Felipe Bianciotto', "Java", 2],
    ['Matias Felipe Bianciotto', "SQL", 2],
    ['Matias Felipe Bianciotto', "Assembler", 2],
    ['Luis Alberto Spinetta', "Assembler", 5],
    ['Cecilia Torales', "Python", 1],
    ['Cecilia Torales', "Java", 1],
    ['Cecilia Torales', "SQL", 1],
    ['María José Ibacache', "Python", 3],
    ['María José Ibacache', "Java", 3],
    ['María José Ibacache', "SQL", 3],
    ['Sam Altman', "Python", 5],
    ['Sam Altman', "Java", 5],
    ['Sam Altman', "SQL", 5],
    ['Sam Altman', "Assembler", 5],
    ['Billie Joe Armstrong', "Python", 1],
    ['Billie Joe Armstrong', "Java", 1],
    ['Billie Joe Armstrong', "SQL", 1],
    ['Billie Joe Armstrong', "Assembler", 1],
    ['Agustina Silva', "Python", 1],
    ['Ivan de Pineda', "SQL", 4],
    ['Ivan de Pineda', "Assembler", 4],
    ['Diego', "Python", 3],
    ['Diego', "Java", 3],
    ['Diego', "SQL", 3],
    ['Lautaro Moreno', "Java", 3],
    ['Lautaro Moreno', "SQL", 3],
    ['mw', "Java", 5],
    ['mw', "SQL", 5],
    ['La Papada de Milei', "Python", 5],
    ['El Timbero Caputo', "SQL", 1],
    ['Pezella', "Assembler", 5],
    ['Llegaron las Pipshas', "Python", 3],
    ['Llegaron las Pipshas', "SQL", 3],
    ['Patricia Bullrich', "Java", 1],
    ['Patricia Bullrich', "SQL", 1],
    ['Patricia Bullrich', "Assembler", 1],
    ['FanDeOrga', "Assembler", 5]
]

#creamos un dataFrame de pandas a partir de los datos y definimos explícitamente los nombres de las columnas del dataframe.  (dataframe=estructura de datos tabular)
df = pd.DataFrame(data, columns=['Nombre', 'Habilidades', 'Nivel Experiencia'])

#preprocesamiento de datos, hay que convertir las habilidades en una representacion numerica, combinamos las habilidades con el nivel de experiencia para crear un conjunto
#binarizar las habilidades
mlb = MultiLabelBinarizer()
habilidades_binarizadas = mlb.fit_transform(df['Habilidades'].apply(lambda x: [x]))

#creamos el data frame de neuvo con las habilidades binarizadasy nivel experiencia
df_final = pd.DataFrame(habilidades_binarizadas, columns=mlb.classes_)
df_final['Nivel Experiencia'] = df['Nivel Experiencia']

#estandarizacion de los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)

#aplicamos k means
kmeans = KMeans(n_clusters=3, random_state=42)
df_final['Cluster'] = kmeans.fit_predict(df_scaled)

#agregamso la columna cluster al dataframe
df['Cluster'] = df_final['Cluster']

#print de los datos asociados a cada cluster
print("Datos asociados a cada cluster:")
for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster])

# aplicamos PCA para reducir la dimensionalidad a 2 componentes
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_scaled)

#creamos un dataframe con los datos reducidos y los clusters
df_pca = pd.DataFrame(reduced_data, columns=['Componente 1', 'Componente 2'])
df_pca['Cluster'] = df_final['Cluster']

#grafico los clusters
plt.figure(figsize=(10, 6))
for cluster in df_pca['Cluster'].unique():
    subset = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(subset['Componente 1'], subset['Componente 2'], label=f'Cluster {cluster}', alpha=0.7)

#grafico centroide
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroides')

#config
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Clusters formados por K-Means (Visualización con PCA)')
plt.legend(title='Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()