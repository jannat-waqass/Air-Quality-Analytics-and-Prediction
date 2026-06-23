"""
Created on Fri Apr 18 11:40:11 2025

@author: Jannat Waqass
"""

# Importing necessary libraries

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Loading the dataset

df = pd.read_csv("C:/Users/Jannat Waqass/Desktop/dataset1.csv")

print(df.info())
# Selecting important features necessary for clustering

features = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
data = df[features].copy()

# Drop rows with missing values in selected features

data.dropna(inplace=True)

#Normalize the data

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#Apply K-Means clustering

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
df = df.loc[data.index]  # Keep only rows used in clustering
df['Cluster'] = clusters

#Analyze cluster centres

centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
print("Cluster Centroids:")
print(centroids)

#Manually Map Clusters to AQI Labels

cluster_labels = {
    0: 'Good',
    1: 'Moderate',
    2: 'Poor',
    3: 'Hazardous'
}
df['Predicted_AQI_Category'] = df['Cluster'].map(cluster_labels)

#Save Labeled Dataset

df.to_csv("C:/Users/Jannat Waqass/Desktop/clustered_dataset1.csv", index=False)

#Visualize the Clusters

sns.set(style='whitegrid')
# Sample if the dataset is large
sample_df = df.sample(n=min(500, len(df)), random_state=42)
sns.pairplot(sample_df[features + ['Cluster']], hue='Cluster', palette='Set2')
plt.suptitle('Air Quality Clusters', y=1.02)
plt.show()


# Load the 2 datasets that need to be integrated

df1 = pd.read_csv("C:/Users/Jannat Waqass/Desktop/clustered_dataset1.csv")
df2 = pd.read_csv("C:/Users/Jannat Waqass/Desktop/dataset2.csv")
print(df1.info())
print(df2.info())
#Remove the missing values

df1.dropna(inplace=True)
df2.dropna(inplace=True)

#Remove duplicate values

df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)

#Remove outliers in both datasets using IQR method

def remove_outliers(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]
    return df
features = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
df1 = remove_outliers(df1, features)
df2 = remove_outliers(df2, ['CO', 'PM2.5', 'NO2'])

# Select and rename columns for consistency

df1_filtered = df1[['CO AQI Value', 'PM2.5 AQI Value', 'NO2 AQI Value', 'Predicted_AQI_Category']]
df1_filtered.columns = ['CO', 'PM2.5', 'NO2', 'Air Quality']
df2_filtered = df2[['CO', 'PM2.5', 'NO2', 'Air Quality']]

# Combine both datasets

combined_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

# Separate features and labels for scaling

features = ['CO', 'PM2.5', 'NO2']
labels = combined_df['Air Quality']

# Normalize features only

scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(combined_df[features]), columns=features)

# Apply PCA (principal component analysis for easier visualization)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
pca_df['Cluster'] = clusters

# Plot PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set2')
plt.title('PCA of Air Quality Data')
plt.show()

# Plot original features
sns.pairplot(combined_df)
plt.suptitle('Pairplot of Air Quality Features', y=1.02)
plt.show()

# Combine scaled features and labels
final_df = pd.concat([scaled_features, labels.reset_index(drop=True)], axis=1)

# Save final integrated dataset
final_df.to_csv("C:/Users/Jannat Waqass/Desktop/integrated_air_quality.csv", index=False)

print("Datasets successfully integrated and saved as 'integrated_air_quality.csv'")

