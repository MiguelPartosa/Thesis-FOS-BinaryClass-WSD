import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


def process_embeddings(df, variance_threshold=0.8):
    """
    Process a dataframe with text columns to generate embeddings, 
    cluster assignments, and PCA components.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with columns: 'Word Sense', 'Verb', 'Usage', 'Is FOS'
    variance_threshold : float, default=0.8
        Proportion of variance to capture with PCA components
    embedding_model : SentenceTransformer, default=None
        Pre-loaded SentenceTransformer model. If None, will load 'all-MiniLM-L6-v2'

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster assignments and PCA components
    """
    # Initialize embedding model if not provided
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    df_embedded = retrieve_embeddings(df, embedding_model)

    # Transform embeddings to DataFrame format
    df_pca_sentence = generate_embedding_df(
        'Sentence Embeddings', df_embedded['Sentence Embeddings'])
    df_pca_verb = generate_embedding_df(
        'Verb Embeddings', df_embedded['Verb Embeddings'])
    df_pca_usage = generate_embedding_df(
        'Usage Embeddings', df_embedded['Usage Embeddings'])
    print("UPDATED")
    df_similarity = pd.DataFrame({'Similarity Scores': df_embedded.apply(lambda row: ComputeSimilarity(
        row['Sentence Embeddings'], row['Usage Embeddings'], embedding_model), axis=1)})

    # Find optimal number of components
    scaler = StandardScaler()
    verb_components = get_components_for_variance(
        df_pca_verb, variance_threshold, scaler)
    usage_components = get_components_for_variance(
        df_pca_usage, variance_threshold, scaler)
    sentence_components = get_components_for_variance(
        df_pca_sentence, variance_threshold, scaler)

    print(f"Number of components for {variance_threshold*100}% variance:")
    print(f"Verb: {verb_components} components")
    print(f"Usage: {usage_components} components")
    print(f"Sentence: {sentence_components} components")

    # Find optimal number of clusters
    verb_k = find_optimal_clusters(df_pca_verb, verb_components, scaler)
    usage_k = find_optimal_clusters(df_pca_usage, usage_components, scaler)
    sentence_k = find_optimal_clusters(
        df_pca_sentence, sentence_components, scaler)

    print(f"Optimal number of clusters:")
    print(f"Verb: {verb_k} clusters")
    print(f"Usage: {usage_k} clusters")
    print(f"Sentence: {sentence_k} clusters")

    # Generate PCA-guided K-means
    final_pca_verb = generate_guided_pca(
        df_pca_verb, verb_components, verb_k, 'Verb', scaler)
    final_pca_usage = generate_guided_pca(
        df_pca_usage, usage_components, usage_k, 'Usage', scaler)
    final_pca_sentence = generate_guided_pca(
        df_pca_sentence, sentence_components, sentence_k, 'Sentence', scaler)

    # Combine results
    final_result = pd.concat(
        [df_similarity, final_pca_usage, final_pca_verb, final_pca_sentence], axis=1)

    return final_result


def get_embeddings(text, model):
    """Generate normalized embeddings for text using the provided model."""
    return model.encode(text, normalize_embeddings=True)


def ComputeSimilarity(embedding1, embedding2, model):
    return model.similarity(embedding1, embedding2)


def retrieve_embeddings(df, model):
    """Generate embeddings for all relevant columns."""
    df_embeddings = df.copy()

    print("Generating embeddings...")
    df_embeddings['Sentence Embeddings'] = df['Word Sense'].apply(
        lambda x: get_embeddings(x, model))
    df_embeddings['Verb Embeddings'] = df['Verb'].apply(
        lambda x: get_embeddings(x, model))
    df_embeddings['Usage Embeddings'] = df['Usage'].apply(
        lambda x: get_embeddings(x, model))

    return df_embeddings


def generate_embedding_df(col_name, embeddings_series):
    """Transform embeddings Series into a DataFrame with one feature per dimension."""
    df = pd.DataFrame()

    for embeddings in tqdm(embeddings_series, desc=f'Transforming {col_name}'):
        new_row_val = []
        new_row_column_name = []

        for idx, embedding_val in enumerate(embeddings):
            new_row_column_name.append(f'{col_name} {idx + 1}')
            new_row_val.append(embedding_val)

        df = pd.concat(
            [df, pd.DataFrame([new_row_val], columns=new_row_column_name)])

    return df.reset_index(drop=True)


def get_explained_variance(df, scaler):
    """Get the explained variance ratio for each principal component."""
    raw_scaled = scaler.fit_transform(df)
    pca = PCA()
    pca.fit(raw_scaled)
    return pca.explained_variance_ratio_


def get_cumulative_variance(df, scaler):
    """Get the cumulative explained variance for principal components."""
    return np.cumsum(get_explained_variance(df, scaler))


def get_components_for_variance(df, threshold, scaler):
    """Find the number of components needed to explain a given proportion of variance."""
    cumulative_var = get_cumulative_variance(df, scaler)
    components_needed = np.argmax(cumulative_var >= threshold) + 1
    return components_needed


def find_optimal_clusters(df, n_components, scaler, max_clusters=30):
    """Find the optimal number of clusters using silhouette analysis."""
    # Reduce dimensions with PCA
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_data)

    # Find optimal k
    best_k, best_score = 2, -1

    # Limit clusters to reasonable number based on data size
    max_k = min(max_clusters, len(df) // 30)  # At least 30 samples per cluster
    max_k = max(max_k, 2)  # At least 2 clusters

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, init="k-means++",
                       n_init=10, random_state=42)
        labels = model.fit_predict(pca_df)

        if len(np.unique(labels)) > 1:
            score = silhouette_score(pca_df, labels)

            if score > best_score:
                best_k, best_score = k, score

    return best_k


def generate_guided_pca(df, components, clusters, title, scaler):
    """Generate PCA components and cluster assignments."""
    # Scale data
    scaled_data = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=components)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_data)

    # Apply KMeans
    kmeans_model = KMeans(n_clusters=clusters,
                          init="k-means++", random_state=42)
    labels = kmeans_model.fit_predict(pca_df)

    # Rename columns
    for i in pca_df.columns:
        pca_df.rename(columns={i: f'{title} component {i+1}'}, inplace=True)

    # Add cluster assignments
    pca_df.insert(0, f'Kmeans {title} Cluster', labels)

    return pca_df


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_excel("path_to_data.xlsx")

    # Process embeddings
    result_df = process_embeddings(df)

    # Save results
    result_df.to_excel("embeddings_clusters_components.xlsx", index=False)

    print("Processing complete. Results saved to 'embeddings_clusters_components.xlsx'")
