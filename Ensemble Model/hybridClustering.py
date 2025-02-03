import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import spacy
import warnings
warnings.filterwarnings('ignore')


class LiteralUsageClassifier:
    def __init__(self, n_topics=10):
        """
        Initialize the classifier with specified number of topics

        Args:
            n_topics (int): Number of topics for LDA
        """
        self.n_topics = n_topics
        # Load spaCy model for advanced text processing
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_text(self, text):
        """
        Preprocess text using spaCy for advanced linguistic features
        """
        doc = self.nlp(text)

        # Extract relevant linguistic features
        tokens = []
        for token in doc:
            # Keep relevant tokens and their linguistic features
            if not token.is_stop and not token.is_punct:
                # Combine token with its POS and dependency
                augmented_token = f"{token.lemma_}_{token.pos_}_{token.dep_}"
                tokens.append(augmented_token)

        return " ".join(tokens)

    def extract_context_features(self, text, target_verb):
        """
        Extract contextual features around the target verb
        """
        doc = self.nlp(text)
        features = []

        for token in doc:
            if token.lemma_.lower() == target_verb.lower():
                # Get surrounding context
                left_context = " ".join([t.text for t in token.lefts])
                right_context = " ".join([t.text for t in token.rights])

                # Get syntactic dependencies
                dependencies = [child.dep_ for child in token.children]

                features.extend([
                    f"left_context_{left_context}",
                    f"right_context_{right_context}",
                    f"dependencies_{'_'.join(dependencies)}"
                ])

        return " ".join(features)

    def fit(self, df):
        """
        Train the classifier using the provided DataFrame

        Args:
            df: DataFrame with columns ['verb to classify', 'literal example use of verb', 
                                      'nonliteral example use of verb']
        """
        # Prepare training data
        literal_examples = []
        nonliteral_examples = []
        verbs = []

        for _, row in df.iterrows():
            verb = row['Verb']
            literal = row['Literal']
            nonliteral = row['Nonliteral']

            # Process literal example
            processed_literal = self.preprocess_text(literal)
            context_features_literal = self.extract_context_features(
                literal, verb)
            literal_examples.append(f"{processed_literal} {
                                    context_features_literal}")
            verbs.append(verb)

            # Process non-literal example
            processed_nonliteral = self.preprocess_text(nonliteral)
            context_features_nonliteral = self.extract_context_features(
                nonliteral, verb)
            nonliteral_examples.append(f"{processed_nonliteral} {
                                       context_features_nonliteral}")
            verbs.append(verb)

        # Combine examples and create labels
        X = literal_examples + nonliteral_examples
        y = [1] * len(literal_examples) + [0] * len(nonliteral_examples)

        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create and fit pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('lda', LatentDirichletAllocation(
                n_components=self.n_topics, random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=100, random_state=42))
        ])

        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Non-literal', 'Literal']))

        return self

    def predict(self, texts, verbs):
        """
        Predict whether new texts contain literal or non-literal usage of verbs

        Args:
            texts (list): List of texts to classify
            verbs (list): List of target verbs to analyze in each text

        Returns:
            list: Predictions (1 for literal, 0 for non-literal)
        """
        processed_texts = []
        for text, verb in zip(texts, verbs):
            processed = self.preprocess_text(text)
            context = self.extract_context_features(text, verb)
            processed_texts.append(f"{processed} {context}")

        return self.pipeline.predict(processed_texts)

    def get_topic_words(self, n_words=10):
        """
        Get the top words for each topic

        Args:
            n_words (int): Number of top words to return per topic

        Returns:
            list: List of top words for each topic
        """
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out(
        )
        topics = []

        for topic_idx, topic in enumerate(self.pipeline.named_steps['lda'].components_):
            top_words = [feature_names[i]
                         for i in topic.argsort()[:-n_words-1:-1]]
            topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")

        return topics


# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'Verb': ['break', 'run', 'burn'],
        'Literal': [
            'He broke the glass by dropping it.',
            'She runs five miles every morning.',
            'The wood burned quickly in the fireplace.'
        ],
        'Nonliteral': [
            'The news broke his heart.',
            'The colors run together in the wash.',
            'He burned with anger at the insult.'
        ]
    }

    df = pd.DataFrame(data)

    # Initialize and train classifier
    classifier = LiteralUsageClassifier(n_topics=5)
    classifier.fit(df)

    # Example predictions
    new_texts = [
        "The branch broke under his weight.",
        "Their friendship broke after the argument.",
        "The machine runs smoothly.",
        "Time runs like water through our fingers."
    ]
    new_verbs = ['break', 'break', 'run', 'run']

    predictions = classifier.predict(new_texts, new_verbs)

    print("\nPredictions for new texts:")
    for text, pred in zip(new_texts, predictions):
        print(f"\nText: {text}")
        print(f"Prediction: {'Literal' if pred == 1 else 'Non-literal'}")
        print(pred)

    print("\nTop words for each topic:")
    topics = classifier.get_topic_words()
    for topic in topics:
        print(topic)
