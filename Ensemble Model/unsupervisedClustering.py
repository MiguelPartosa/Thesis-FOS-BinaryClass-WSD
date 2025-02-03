import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import spacy
import warnings
warnings.filterwarnings('ignore')


class UnsupervisedLiteralClassifier:
    def __init__(self, n_topics=10, n_clusters=2):
        self.n_topics = n_topics
        self.n_clusters = n_clusters
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_for_features(self, text):
        """
        Preprocess text for classification features
        """
        doc = self.nlp(text)
        features = []
        for token in doc:
            if not token.is_stop and not token.is_punct:
                features.extend([
                    f"{token.lemma_}_{token.pos_}",
                    f"dep_{token.dep_}",
                    f"head_{token.head.lemma_}"
                ])
        return " ".join(features)

    def preprocess_for_topics(self, text):
        """
        Preprocess text for topic modeling - keeping actual words
        """
        doc = self.nlp(text)
        # Keep meaningful words for topic modeling
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct
                  and token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        return " ".join(tokens)

    def fit(self, df):
        """
        Fit the unsupervised model
        """
        # Prepare examples for classification
        processed_features = []
        # Prepare examples for topic modeling
        processed_topics = []

        for _, row in df.iterrows():
            # Process for classification features
            literal_features = self.preprocess_for_features(
                row['literal example use of verb'])
            nonliteral_features = self.preprocess_for_features(
                row['nonliteral example use of verb'])
            processed_features.extend([literal_features, nonliteral_features])

            # Process for topic modeling
            literal_topics = self.preprocess_for_topics(
                row['literal example use of verb'])
            nonliteral_topics = self.preprocess_for_topics(
                row['nonliteral example use of verb'])
            processed_topics.extend([literal_topics, nonliteral_topics])

        # Create feature matrix for classification
        self.feature_vectorizer = TfidfVectorizer(max_features=5000)
        feature_matrix = self.feature_vectorizer.fit_transform(
            processed_features)

        # Create separate matrix for topic modeling
        self.topic_vectorizer = TfidfVectorizer(max_features=1000)
        topic_matrix = self.topic_vectorizer.fit_transform(processed_topics)

        # Apply topic modeling
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42
        )
        self.topic_matrix = self.lda.fit_transform(topic_matrix)

        # Apply clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42
        )
        self.clusters = self.kmeans.fit_predict(feature_matrix)

        # Analyze clusters
        self.analyze_clusters()

        return self

    def analyze_clusters(self):
        """
        Analyze characteristics of each cluster
        """
        self.cluster_profiles = {}

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.clusters == cluster_id
            cluster_docs = self.topic_matrix[cluster_mask]

            # Get average topic distribution for cluster
            avg_topics = np.mean(cluster_docs, axis=0)

            # Store dominant topics
            dominant_topics = np.argsort(-avg_topics)[:3]

            self.cluster_profiles[cluster_id] = {
                'size': np.sum(cluster_mask),
                'dominant_topics': dominant_topics,
                'topic_distribution': avg_topics
            }

    def get_topic_words(self, n_words=10):
        """
        Get the top words for each topic
        """
        feature_names = self.topic_vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i]
                         for i in topic.argsort()[:-n_words-1:-1]]
            topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")

        return topics

    def predict(self, texts):
        """
        Predict cluster assignments for new texts
        """
        # Process texts for classification
        processed_texts = [
            self.preprocess_for_features(text) for text in texts]

        # Transform texts to TF-IDF
        feature_matrix = self.feature_vectorizer.transform(processed_texts)

        # Predict clusters
        predictions = self.kmeans.predict(feature_matrix)

        # Get prediction confidence
        distances = self.kmeans.transform(feature_matrix)
        confidences = 1 / (1 + distances)

        return predictions, confidences


# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'verb to classify': ['break', 'run', 'burn', 'fall'],
        'literal example use of verb': [
            'He broke the glass by dropping it.',
            'She runs five miles every morning.',
            'The wood burned quickly in the fireplace.',
            'The apple fell from the tree.'
        ],
        'nonliteral example use of verb': [
            'The news broke his heart.',
            'The colors run together in the wash.',
            'He burned with anger at the insult.',
            'She fell in love instantly.'
        ]
    }

    df = pd.DataFrame(data)

    # Initialize and fit classifier
    classifier = UnsupervisedLiteralClassifier(n_topics=5)
    classifier.fit(df)

    # Print topics
    print("\nDiscovered Topics:")
    topics = classifier.get_topic_words()
    for topic in topics:
        print(topic)

    # Example predictions
    new_texts = [
        "The branch broke under his weight.",
        "Their friendship broke after the argument.",
        "The machine runs smoothly.",
        "Time runs like water through our fingers."
    ]

    predictions, confidences = classifier.predict(new_texts)

    print("\nPredictions for new texts:")
    for text, pred, conf in zip(new_texts, predictions, confidences):
        print(f"\nText: {text}")
        print(f"Predicted cluster: {pred}")
        print(f"Confidence scores: {conf}")
