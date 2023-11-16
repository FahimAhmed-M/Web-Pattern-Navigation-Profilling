import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample dataset (replace this with your actual dataset)
data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'page_views': [10, 15, 8, 20, 12, 18, 25, 30, 22, 28],
    'time_spent': [30, 45, 20, 60, 35, 50, 70, 80, 60, 75],
    'clicks': [5, 8, 3, 10, 6, 9, 12, 15, 11, 14]
}

df = pd.DataFrame(data)

# Feature scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['page_views', 'time_spent', 'clicks']])

# Clustering users based on behavior
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Visualization
plt.figure(figsize=(10, 6))

for cluster in range(num_clusters):
    plt.scatter(df[df['cluster'] == cluster]['page_views'],
                df[df['cluster'] == cluster]['time_spent'],
                label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Cluster Centers')

plt.xlabel('Page Views')
plt.ylabel('Time Spent (seconds)')
plt.title('User Clusters Based on Behavior')
plt.legend()
plt.show()

# User segmentation based on clusters
user_segments = {}
for cluster in range(num_clusters):
    segment = df[df['cluster'] == cluster][['user_id', 'page_views', 'time_spent', 'clicks']]
    user_segments[f'Segment {cluster + 1}'] = segment

# Display user segments
for segment, users in user_segments.items():
    print(f"\n{segment}:\n{users}\n")
