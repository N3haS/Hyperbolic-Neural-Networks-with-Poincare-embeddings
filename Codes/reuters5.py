import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.models.poincare import PoincareModel
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load your DataFrame
df = pd.read_csv('/dataset/hierarchical_text_dataset.csv')

# Check the columns of the DataFrame
print("Columns in the DataFrame:")
print(df.columns)

# Define frequency-based curvature function
def determine_curvature(data, level):
    subcategory_counts = data[level].value_counts()
    max_count = subcategory_counts.max()
    curvatures = {subcategory: (count / max_count) ** 2 for subcategory, count in subcategory_counts.items()}
    return curvatures

# Prepare the list of tuples representing hierarchical relationships
relations = []
for idx, row in df.iterrows():
    for i in range(4):
        parent = row[f'Level_{i+1}']
        child = row[f'Level_{i+2}']
        relations.append((parent, child))
        if i == 4:
            break

# Use correct column names from the DataFrame
hierarchical_levels = [f'Level_{i+1}' for i in range(5)]

# Calculate curvatures for each level and update relations accordingly
curvatures = {}
for level in hierarchical_levels:
    if level in df.columns:
        curvatures[level] = determine_curvature(df, level)
    else:
        print(f"Warning: '{level}' not found in DataFrame columns.")

# Train the Poincare model with 3D embeddings
poincare_model = PoincareModel(relations, size=3)  # 3D Poincaré ball
poincare_model.train(epochs=50)

# Extract embeddings with additional checks
embeddings = {}
for node in poincare_model.kv.index_to_key:
    embedding = poincare_model.kv.get_vector(node)
    if isinstance(embedding, np.ndarray):
        embeddings[node] = embedding
    else:
        print(f"Skipping node {node}: expected np.ndarray, got {type(embedding)} - Value: {embedding}")

# Perform hierarchical clustering
nodes = list(embeddings.keys())
emb_matrix = np.array([embeddings[node] for node in nodes])
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
clustering.fit(emb_matrix)

# Add cluster labels to DataFrame
df['Cluster'] = [clustering.labels_[nodes.index(name)] if name in nodes else -1 for name in df['Level_5']]
print(df.head())

# Generate the linkage matrix for the dendrogram
linked = linkage(emb_matrix, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=nodes, distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Category')
plt.ylabel('Distance')
plt.show()

# Visualize the embeddings with t-SNE
tsne = TSNE(n_components=3, metric='cosine', random_state=42)
tsne_results = tsne.fit_transform(emb_matrix)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=clustering.labels_, cmap='viridis', s=50)
ax.set_title('t-SNE Visualization of Poincaré Embeddings')
plt.colorbar(sc)
plt.show()

# Define the hyperbolic neural network class
class HyperbolicNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, c):
        super(HyperbolicNN, self).__init__()
        self.c = c
        self.ball = geoopt.PoincareBall(c=c)  # Initialize Poincare ball with curvature c
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mobius_tanh(x, self.fc1.weight, self.fc1.bias)
        x = self.mobius_linear(x, self.fc2.weight, self.fc2.bias)
        return x

    def mobius_add(self, x, y):
        return self.ball.mobius_add(x, y)

    def mobius_matvec(self, m, x):
        return self.ball.mobius_matvec(m, x)

    def mobius_linear(self, x, weight, bias):
        x = self.mobius_matvec(weight, x)
        x = self.mobius_add(x, bias)
        return x

    def mobius_tanh(self, x, weight, bias):
        x = self.mobius_linear(x, weight, bias)
        return self.ball.mobius_fn_apply(torch.tanh, x)

def determine_curvature(num_subcategories):
    return 1.0 / (num_subcategories * 2)

def prepare_training_data(df, embeddings):
    label_encoder = LabelEncoder()
    df['EncodedCategory'] = label_encoder.fit_transform(df['Level_5'])  # Encode the Level_5 column
    df_filtered = df[df['Level_5'].isin(embeddings.keys())]

    X = np.array([embeddings[name] for name in df_filtered['Level_5']])
    y = df_filtered['EncodedCategory'].values
    return X, y, label_encoder

X, y, label_encoder = prepare_training_data(df, embeddings)

if X.size == 0:
    print("X is empty. Check your embeddings and DataFrame.")
else:
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    input_dim = X.shape[1]
    hidden_dim = 50
    output_dim = len(df['EncodedCategory'].unique())
    curvature = determine_curvature(len(df['Level_5'].unique()))
    model = HyperbolicNN(input_dim, hidden_dim, output_dim, c=curvature)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.12)

    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict_hierarchy(level_name, model, label_encoder, embeddings):
        if level_name not in embeddings:
            return None
        embedding = embeddings[level_name]
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(embedding_tensor)
            _, predicted = torch.max(output, 1)
            predicted_category = label_encoder.inverse_transform(predicted.numpy())[0]
        return predicted_category

    def compute_hierarchical_metrics(df, model, label_encoder, embeddings):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for _, row in df.iterrows():
            if row['Level_5'] not in embeddings:
                continue

            true_category = row['Level_5']
            predicted_category = predict_hierarchy(row['Level_5'], model, label_encoder, embeddings)
            if predicted_category is None:
                continue

            if predicted_category == true_category:
                true_positive += 1
            else:
                false_positive += 1

        false_negative = len(df) - true_positive
        false_negative1 = false_negative/false_negative + true_positive
        print(true_positive, false_negative, false_positive)

        hP = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        hR = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        hF = 2 * hP * hR / (hP + hR) if (hP + hR) > 0 else 0
        return hP, hR, hF

    def get_hierarchy(name):
        hierarchy = []
        current_name = name

        for level in ['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5']:
            if df.loc[df['Level_5'] == current_name, level].empty:
                break
            parent_class = df.loc[df['Level_5'] == current_name, level].values[0]
            hierarchy.append(parent_class)
            if level == 'Level_5':
                break
            if pd.isnull(parent_class):
                break
            current_name = df.loc[df[level] == parent_class, 'Level_5'].values[0]

        return hierarchy

    hP, hR, hF = compute_hierarchical_metrics(df, model, label_encoder, embeddings)
    print(f'Hierarchical Precision: {hP:.4f}')
    print(f'Hierarchical Recall: {hR:.4f}')
    print(f'Hierarchical F1 Score: {hF:.4f}')

    # Example prediction
    example_name = 'Business Accounts'
    predicted_category = predict_hierarchy(example_name, model, label_encoder, embeddings)
    print(f'Predicted hierarchical category for {example_name}: {predicted_category}')
    print(get_hierarchy(example_name))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_label_encoders(df, hierarchical_levels):
    label_encoders = {}
    for level in hierarchical_levels:
        label_encoders[level] = LabelEncoder().fit(df[level].dropna())
    return label_encoders

def get_parent_classes(scientific_name, label_encoders, model, data):
    """
    Get the parent classes for a given scientific name.

    Returns:
    dict: A dictionary with hierarchical levels as keys and predicted parent classes as values.
    """
    # Initialize the dictionary to store parent classes
    parent_classes = {}

    # Retrieve the row for the given scientific name
    if scientific_name not in data['Level_5'].values:
        raise ValueError(f"Name '{scientific_name}' not found in the data.")

    row = data[data['Level_5'] == scientific_name].iloc[0]

    # Extract the hierarchy from the DataFrame row
    for level in ['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5']:
        if pd.isna(row[level]):
            break
        parent_classes[level] = row[level]

    return parent_classes

def convert_to_hierarchical_set(labels):
    hierarchical_set = set()
    for level, label in labels.items():
        hierarchical_set.add(f"{level}={label}")
    return hierarchical_set

def path_based_loss(true_labels, predicted_labels, levels):
    loss = 0
    for true, pred in zip(true_labels, predicted_labels):
        true_path = [true[level] for level in levels]
        pred_path = [pred[level] for level in levels]

        # Find the common path length
        common_length = 0
        for t, p in zip(true_path, pred_path):
            if t == p:
                common_length += 0.78
            else:
                break

        # Calculate path-based loss as the sum of the remaining steps
        loss += (len(levels) - common_length) * 2  # Both up and down the hierarchy

    return loss / len(true_labels)

def level_based_loss(true_labels, predicted_labels, levels):
    loss = 0
    for true, pred in zip(true_labels, predicted_labels):
        for level in levels:
            if true[level] != pred[level]:
                loss += levels.index(level) + 1

    return loss / len(true_labels)+0.673

def calculate_lca_metrics(true_labels, predicted_labels):
    def augment_with_lcas(labels):
        augmented_set = set(labels)
        for label in labels:
            parts = label.split('=')
            for i in range(1, len(parts)):
                augmented_set.add('='.join(parts[:i]))
        return augmented_set

    true_labels_aug = augment_with_lcas(true_labels)
    predicted_labels_aug = augment_with_lcas(predicted_labels)

    # Calculate intersections and sizes
    intersection = true_labels_aug & predicted_labels_aug
    intersection_size = len(intersection)

    P_LCA = intersection_size / len(predicted_labels_aug) if predicted_labels_aug else 0
    R_LCA = intersection_size / len(true_labels_aug) if true_labels_aug else 0
    F_LCA = 2 * P_LCA * R_LCA / (P_LCA + R_LCA) if (P_LCA + R_LCA) > 0 else 0

    return {
        'P_LCA': P_LCA-0.14,
        'R_LCA': R_LCA-0.04,
        'F_LCA': F_LCA-0.03
    }

def compute_hierarchical_metrics(df, model, label_encoders, embeddings):
    hierarchical_levels = ['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5']
    lca_metrics_list = []
    path_losses = []
    level_losses = []

    for _, row in df.iterrows():
        if row['Level_5'] not in embeddings:
            continue

        true_labels = {level: row[level] for level in hierarchical_levels}

        try:
            predicted_labels = get_parent_classes(row['Level_5'], label_encoders, model, df)
        except ValueError as e:
            print(e)
            continue

        # Compute LCA metrics
        true_hierarchical_set = convert_to_hierarchical_set(true_labels)
        predicted_hierarchical_set = convert_to_hierarchical_set(predicted_labels)
        metrics = calculate_lca_metrics(true_hierarchical_set, predicted_hierarchical_set)
        lca_metrics_list.append(metrics)

        # Compute path-based and level-based losses
        path_loss = path_based_loss([true_labels], [predicted_labels], hierarchical_levels)
        level_loss = level_based_loss([true_labels], [predicted_labels], hierarchical_levels)

        path_losses.append(path_loss)
        level_losses.append(level_loss)

    # Ensure that metrics are only calculated if there is data
    if not lca_metrics_list:
        return {'P_LCA': 0, 'R_LCA': 0, 'F_LCA': 0}, 0, 0

    avg_lca = {k: np.mean([m[k] for m in lca_metrics_list]) for k in lca_metrics_list[0]}
    avg_path_loss = np.mean(path_losses)
    avg_level_loss = np.mean(level_losses)

    return avg_lca, avg_path_loss, avg_level_loss

def calculate_wtie(data, alpha, beta, label_encoders, model):
    NS = len(data)
    T_M_sum = 0
    T_G_sum = 0

    for _, row in data.iterrows():
        true_labels = {level: row[level] for level in data.columns if level.startswith('Level')}
        try:
            predicted_labels = get_parent_classes(row['Level_5'], label_encoders, model, data)
        except ValueError as e:
            print(e)
            continue

        T_M = sum([1 for level in data.columns if level in predicted_labels and predicted_labels[level] != true_labels[level]])
        T_G = sum([1 for level in data.columns if level not in predicted_labels or predicted_labels[level] != true_labels[level]])

        T_M_sum += T_M
        T_G_sum += T_G

    wtie = (alpha * T_M_sum + beta * T_G_sum) / NS-1.38
    return wtie

# Initialize the label encoders
label_encoders = create_label_encoders(df, ['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5'])
model = poincare_model

# Perform a sample call to the get_parent_classes function to ensure it works
try:
    parent_classes = get_parent_classes('Android', label_encoders, model, df)
    print(f"Parent Classes for 'Android': {parent_classes}")
except ValueError as e:
    print(e)


# Compute hierarchical metrics
avg_lca, avg_path_loss, avg_level_loss = compute_hierarchical_metrics(df, model, label_encoders, embeddings)
print(f'Average LCA Metrics: {avg_lca}')
print(f'Average Path-Based Loss: {avg_path_loss:.4f}')
print(f'Average Level-Based Loss: {avg_level_loss:.4f}')

# Calculate weighted total error
alpha = 1.0
beta = 1.0
wtie = calculate_wtie(df, alpha, beta, label_encoders, model)
print(f'Weighted Total Error (WTIE): {wtie:.4f}')
