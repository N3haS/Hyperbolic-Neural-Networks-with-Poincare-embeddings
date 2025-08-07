import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.poincare import PoincareModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load your DataFrame
df = pd.read_csv('/dataset/taxonomy.csv', encoding = 'latin1') 
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Prepare the list of tuples representing hierarchical relationships
relations = []
for idx, row in df.iterrows():
    for i in range(6):
        parent = row[i]
        child = row[i+1]
        relations.append((parent, child))
        if i == 5:
            break

# Train the Poincare model with 3D embeddings
poincare_model = PoincareModel(relations, size=3)  # 3D Poincaré ball
poincare_model.train(epochs=50)

# Get embeddings
embeddings = {node: poincare_model.kv[node] for node in poincare_model.kv.key_to_index}

# Perform hierarchical clustering
nodes = list(embeddings.keys())
emb_matrix = np.array([embeddings[node] for node in nodes])
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
clustering.fit(emb_matrix)

# Add cluster labels to DataFrame
df['Cluster'] = [clustering.labels_[nodes.index(name)] if name in nodes else -1 for name in df['Scientific.Name']]
print(df)

# Generate the linkage matrix for the dendrogram
linked = linkage(emb_matrix, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=nodes, distance_sort='descending', show_leaf_counts=True)
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
        # Perform Mobius addition using the Poincare ball model
        return self.ball.mobius_add(x, y)

    def mobius_matvec(self, m, x):
        # Perform Mobius matrix-vector multiplication using the Poincare ball model
        return self.ball.mobius_matvec(m, x)

    def mobius_linear(self, x, weight, bias):
        # Perform Mobius linear transformation
        x = self.mobius_matvec(weight, x)
        x = self.mobius_add(x, bias)
        return x

    def mobius_tanh(self, x, weight, bias):
        # Apply Mobius tanh activation
        x = self.mobius_linear(x, weight, bias)
        return self.ball.mobius_fn_apply(torch.tanh, x)

def determine_curvature(num_subcategories):
    # Choose curvature value
    return 1.0 / (num_subcategories * 2)

def prepare_training_data(df, embeddings):
    label_encoder = LabelEncoder()
    df['EncodedGenus'] = label_encoder.fit_transform(df['Genus'])
    df_filtered = df[df['Scientific.Name'].isin(embeddings.keys())]

    # Convert embeddings into input matrix X and target labels y
    X = np.array([embeddings[name] for name in df_filtered['Scientific.Name']])
    y = df_filtered['EncodedGenus'].values
    return X, y, label_encoder

# Prepare the data
X, y, label_encoder = prepare_training_data(df, embeddings)

# Check if df_filtered is empty
if X.size == 0:
    print("X is empty. Check your embeddings and DataFrame.")
else:
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Train the Hyperbolic neural network with adaptive curvature
    input_dim = X.shape[1]
    hidden_dim = 50
    output_dim = len(df['EncodedGenus'].unique())
    curvature = determine_curvature(len(df['Genus'].unique()))
    model = HyperbolicNN(input_dim, hidden_dim, output_dim, c=curvature)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.09)

    epochs = 1500
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict_hierarchy(scientific_name, model, label_encoder, embeddings):
        embedding = embeddings[scientific_name]
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(embedding_tensor)
            _, predicted = torch.max(output, 1)
            predicted_genus = label_encoder.inverse_transform(predicted.numpy())[0]
        return predicted_genus

    def compute_hierarchical_metrics(df, model, label_encoder, embeddings):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for _, row in df.iterrows():
            if row['Scientific.Name'] not in embeddings:
                continue

            true_genus = row['Genus']
            predicted_genus = predict_hierarchy(row['Scientific.Name'], model, label_encoder, embeddings)
            if predicted_genus == true_genus:
                true_positive += 1
            else:
                false_positive += 1
            false_negative = len(df) - true_positive
            #print(true_positive, false_negative,false_positive)
            false_negative1 = false_negative/false_negative+true_positive

        hP = true_positive / (true_positive + false_positive)
        hR = true_positive / (true_positive + false_negative1)
        hF = 2 * hP * hR / (hP + hR)
        return hP, hR, hF

    def predict_hierarchy1(scientific_name, model, label_encoder, embeddings, max_levels=6):
        predicted_hierarchy1 = []
        current_name = scientific_name

        for _ in range(max_levels):
            if current_name not in embeddings:
                break

            embedding = embeddings[current_name]
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(embedding_tensor)
                _, predicted = torch.max(output, 1)
                predicted_genus = label_encoder.inverse_transform(predicted.numpy())[0]

                predicted_hierarchy1.append(predicted_genus)

            # Move up the hierarchy if available
            if current_name in df['Scientific.Name'].values:
                parent_name = df[df['Scientific.Name'] == current_name]['Genus'].values[0]
                if pd.isnull(parent_name):
                    break
                current_name = parent_name
        return predicted_hierarchy1

    def get_hierarchy(name):
        hierarchy = []
        current_name = name

        for level in ['Scientific.Name', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']:
            parent_class = df.loc[df['Scientific.Name'] == current_name, level].values[0]
            hierarchy.append(parent_class)
            if level == 'Kingdom':
                break
            if pd.isnull(parent_class):
                print(f"Parent class is null for '{current_name}' at level '{level}'. Terminating hierarchy retrieval.")
                break
            current_name = df.loc[df[level] == parent_class, 'Scientific.Name'].values[0]

        return hierarchy

    hP, hR, hF = compute_hierarchical_metrics(df, model, label_encoder, embeddings)
    print(f'Hierarchical Precision: {hP:.4f}')
    print(f'Hierarchical Recall: {hR:.4f}')
    print(f'Hierarchical F1 Score: {hF:.4f}')

    # Example prediction for 'Canis Lupus'
    example_name = 'Canis Lupus'
    predicted_hierarchy = predict_hierarchy(example_name, model, label_encoder, embeddings)
    print(f"Predicted Hierarchy for '{example_name}': {predicted_hierarchy}")

    example_name = 'Xenopus laevis'
    hierarchy = get_hierarchy(example_name)
    print(f"Hierarchy for '{example_name}': {hierarchy}")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import geoopt
from sklearn.preprocessing import LabelEncoder

def create_label_encoders(df, hierarchical_levels):
    label_encoders = {}
    for level in hierarchical_levels:
        label_encoders[level] = LabelEncoder().fit(df[level].dropna())
    return label_encoders

def get_parent_classes(scientific_name, label_encoders, model, data):
    """
    Get the parent classes for a given scientific name.

    Parameters:
    scientific_name (str): The scientific name for which to get parent classes.
    label_encoders (dict): Dictionary of LabelEncoders for each hierarchical level.
    model: Trained model for hierarchical prediction (if needed).
    data (pd.DataFrame): DataFrame containing the hierarchical taxonomy information.

    Returns:
    dict: A dictionary with hierarchical levels as keys and predicted parent classes as values.
    """
    # Initialize the dictionary to store parent classes
    parent_classes = {}

    # Retrieve the row for the given scientific name
    if scientific_name not in data['Scientific.Name'].values:
        raise ValueError(f"Scientific name '{scientific_name}' not found in the data.")

    row = data[data['Scientific.Name'] == scientific_name].iloc[0]

    # Extract the hierarchy from the DataFrame row
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']:
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
                common_length += 1
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

    return loss / len(true_labels)

def calculate_lca_metrics(true_labels, predicted_labels):
    def augment_with_lcas(labels):
        augmented_set = set(labels)
        for label in labels:
            parts = label.split('.')
            for i in range(1, len(parts)):
                augmented_set.add('.'.join(parts[:i]))
        return augmented_set

    true_labels_aug = augment_with_lcas(true_labels)
    predicted_labels_aug = augment_with_lcas(predicted_labels)

    # Calculate intersections and sizes
    intersection = true_labels_aug & predicted_labels_aug
    intersection_size = len(intersection)

    P_LCA = intersection_size / len(predicted_labels_aug)
    R_LCA = intersection_size / len(true_labels_aug)
    F_LCA = 2 * P_LCA * R_LCA / (P_LCA + R_LCA) if (P_LCA + R_LCA) > 0 else 0

    return {
        'P_LCA': P_LCA,
        'R_LCA': R_LCA,
        'F_LCA': F_LCA
    }

def compute_hierarchical_metrics(df, model, label_encoders, embeddings):
    hierarchical_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    lca_metrics_list = []
    path_losses = []
    level_losses = []

    for _, row in df.iterrows():
        if row['Scientific.Name'] not in embeddings:
            continue

        true_labels = {level: row[level] for level in hierarchical_levels}
        predicted_labels = get_parent_classes(row['Scientific.Name'], label_encoders, model, df)

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

    avg_lca = {k: np.mean([m[k] for m in lca_metrics_list]) for k in lca_metrics_list[0]}
    avg_path_loss = np.mean(path_losses)
    avg_level_loss = np.mean(level_losses)

    return avg_lca, avg_path_loss, avg_level_loss

def calculate_wtie(data, alpha, beta):
    NS = len(data)
    T_M_sum = 0
    T_G_sum = 0

    for _, row in data.iterrows():
        true_labels = {level: row[level] for level in data.columns}
        predicted_labels = get_parent_classes(row['Scientific.Name'], label_encoders, model, data)

        T_M = sum([1 for level in data.columns if level in predicted_labels and predicted_labels[level] != true_labels[level]])
        T_G = sum([1 for level in data.columns if level not in predicted_labels or predicted_labels[level] != true_labels[level]])

        T_M_sum += T_M
        T_G_sum += T_G

    wtie = (alpha * T_M_sum + beta * T_G_sum) / NS
    return wtie

# Example usage:
df = pd.read_csv('taxonomy.csv', encoding='latin1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Load or define your model and embeddings here
# model = ...
# embeddings = ...

label_encoders = create_label_encoders(df, ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])

avg_lca, avg_path_loss, avg_level_loss = compute_hierarchical_metrics(df, model, label_encoders, embeddings)
print(f'Average LCA Metrics: {avg_lca}')
print(f'Average Path-Based Loss: {avg_path_loss:.4f}')
print(f'Average Level-Based Loss: {avg_level_loss:.4f}')

# Calculate weighted total error
alpha = 1.0
beta = 1.0
wtie = calculate_wtie(df, alpha, beta)
print(f'Weighted Total Error (WTIE): {wtie:.4f}')

