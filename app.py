#!/usr/bin/env python
"""
Text Representation Learning App
===============================

This app demonstrates two key methods for converting text to numerical vectors:
1. TF-IDF (Term Frequency-Inverse Document Frequency)
2. Word Embeddings (Word2Vec)

Each method is implemented with step-by-step explanations and examples.
"""

import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from gensim.models import Word2Vec
import re
import warnings
warnings.filterwarnings('ignore')

class TFIDFCalculator:
    """
    A simple TF-IDF calculator to demonstrate the concepts step by step.
    """
    
    def __init__(self):
        self.vocabulary = set()
        self.documents = []
        
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def add_documents(self, documents):
        """Add documents to our corpus"""
        self.documents = []
        for doc in documents:
            tokens = self.preprocess_text(doc)
            self.documents.append(tokens)
            self.vocabulary.update(tokens)
        
        print(f"📚 Added {len(documents)} documents to corpus")
        print(f"🔤 Vocabulary size: {len(self.vocabulary)}")
        
    def calculate_tf(self, document, term):
        """Calculate Term Frequency for a specific term in a document"""
        term_count = document.count(term)
        total_terms = len(document)
        tf = term_count / total_terms if total_terms > 0 else 0
        
        print(f"  TF('{term}'): {term_count}/{total_terms} = {tf:.4f}")
        return tf
    
    def calculate_idf(self, term):
        """Calculate Inverse Document Frequency for a term"""
        docs_containing_term = sum(1 for doc in self.documents if term in doc)
        total_docs = len(self.documents)
        idf = math.log(total_docs / docs_containing_term) if docs_containing_term > 0 else 0
        
        print(f"  IDF('{term}'): log({total_docs}/{docs_containing_term}) = {idf:.4f}")
        return idf
    
    def calculate_tfidf(self, doc_index, term):
        """Calculate TF-IDF score for a term in a specific document"""
        if doc_index >= len(self.documents):
            return 0
            
        document = self.documents[doc_index]
        
        print(f"\n🔍 Calculating TF-IDF for '{term}' in Document {doc_index + 1}")
        tf = self.calculate_tf(document, term)
        idf = self.calculate_idf(term)
        tfidf = tf * idf
        
        print(f"  TF-IDF('{term}'): {tf:.4f} × {idf:.4f} = {tfidf:.4f}")
        return tfidf
    
    def get_tfidf_vector(self, doc_index):
        """Get TF-IDF vector for a document"""
        if doc_index >= len(self.documents):
            return {}
            
        vector = {}
        for term in self.vocabulary:
            vector[term] = self.calculate_tfidf(doc_index, term)
        
        return vector


class WordEmbeddingDemo:
    """
    Demonstrates Word Embeddings using Word2Vec
    """
    
    def __init__(self):
        self.model = None
        self.pretrained_model = None
        
    def train_word2vec(self, sentences):
        """Train a simple Word2Vec model on our data"""
        # Preprocess sentences
        processed_sentences = []
        for sentence in sentences:
            tokens = re.sub(r'[^\w\s]', '', sentence.lower()).split()
            processed_sentences.append(tokens)
        
        print("🧠 Training Word2Vec model...")
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=processed_sentences,
            vector_size=50,  # Small vector size for demo
            window=3,        # Context window
            min_count=1,     # Include all words
            workers=1,       # Single thread for reproducibility
            sg=1            # Skip-gram model
        )
        
        print(f"✅ Model trained! Vocabulary size: {len(self.model.wv.key_to_index)}")
        
    def load_pretrained_model(self):
        """Load a pre-trained model (this may take time on first run)"""
        try:
            print("📥 Loading pre-trained Word2Vec model (this may take a moment)...")
            # Using a smaller model for demo purposes
            self.pretrained_model = api.load('word2vec-google-news-300')
            print("✅ Pre-trained model loaded!")
            return True
        except Exception as e:
            print(f"❌ Could not load pre-trained model: {e}")
            print("💡 You can still use the custom trained model!")
            return False
    
    def get_word_vector(self, word, use_pretrained=False):
        """Get vector representation of a word"""
        model = self.pretrained_model if use_pretrained and self.pretrained_model else self.model
        
        if model and word in model.wv:
            return model.wv[word]
        else:
            print(f"❌ Word '{word}' not found in vocabulary")
            return None
    
    def find_similar_words(self, word, use_pretrained=False, topn=5):
        """Find words similar to the given word"""
        model = self.pretrained_model if use_pretrained and self.pretrained_model else self.model
        
        if model and word in model.wv:
            similar = model.wv.most_similar(word, topn=topn)
            print(f"\n🔍 Words most similar to '{word}':")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")
            return similar
        else:
            print(f"❌ Word '{word}' not found in vocabulary")
            return []
    
    def word_analogy(self, word1, word2, word3, use_pretrained=False):
        """Perform word analogy: word1 is to word2 as word3 is to ?"""
        model = self.pretrained_model if use_pretrained and self.pretrained_model else self.model
        
        if not model:
            print("❌ No model available")
            return None
            
        try:
            # Calculate: vector(word2) - vector(word1) + vector(word3)
            result = model.wv.most_similar(
                positive=[word2, word3],
                negative=[word1],
                topn=1
            )
            
            answer = result[0][0]
            confidence = result[0][1]
            
            print(f"\n🧮 Word Analogy: '{word1}' is to '{word2}' as '{word3}' is to '{answer}'")
            print(f"   Confidence: {confidence:.4f}")
            
            return answer
            
        except Exception as e:
            print(f"❌ Could not compute analogy: {e}")
            return None


def demonstrate_tfidf():
    """Demonstrate TF-IDF with examples"""
    print("=" * 60)
    print("🔢 TF-IDF DEMONSTRATION")
    print("=" * 60)
    
    # Sample documents
    documents = [
        "The dog is cute and playful",
        "My dog is a good boy who loves treats",
        "Cats are independent and mysterious animals",
        "The cute cat sleeps all day"
    ]
    
    print("📄 Sample Documents:")
    for i, doc in enumerate(documents):
        print(f"  Doc {i+1}: \"{doc}\"")
    
    # Initialize our TF-IDF calculator
    tfidf_calc = TFIDFCalculator()
    tfidf_calc.add_documents(documents)
    
    # Demonstrate manual calculation
    print("\n" + "="*40)
    print("MANUAL TF-IDF CALCULATION")
    print("="*40)
    
    # Calculate TF-IDF for specific terms
    test_terms = ["cute", "dog", "the"]
    for term in test_terms:
        tfidf_calc.calculate_tfidf(0, term)  # Calculate for first document
    
    # Compare with sklearn implementation
    print("\n" + "="*40)
    print("SKLEARN TF-IDF COMPARISON")
    print("="*40)
    
    # Reconstruct original documents for sklearn
    original_docs = [" ".join(doc) for doc in tfidf_calc.documents]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(original_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"📊 TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"🔤 Features: {list(feature_names)}")
    
    # Show TF-IDF scores for first document
    print(f"\n📋 TF-IDF scores for Document 1:")
    doc_tfidf = tfidf_matrix[0].toarray()[0]
    for i, score in enumerate(doc_tfidf):
        if score > 0:
            print(f"  {feature_names[i]}: {score:.4f}")


def demonstrate_word_embeddings():
    """Demonstrate Word Embeddings"""
    print("\n\n" + "=" * 60)
    print("🧠 WORD EMBEDDINGS DEMONSTRATION")
    print("=" * 60)
    
    # Sample sentences for training
    sentences = [
        "The king rules the kingdom with wisdom",
        "The queen is wise and strong",
        "A man walked down the street",
        "A woman bought groceries at the store", 
        "Paris is the capital of France",
        "Berlin is the capital of Germany",
        "London is a beautiful city",
        "Dogs are loyal pets that love their owners",
        "Cats are independent animals",
        "The cute puppy played in the garden"
    ]
    
    print("📚 Training sentences:")
    for i, sentence in enumerate(sentences[:5]):  # Show first 5
        print(f"  {i+1}. \"{sentence}\"")
    print(f"  ... and {len(sentences)-5} more sentences")
    
    # Initialize word embedding demo
    embedding_demo = WordEmbeddingDemo()
    
    # Train our own model
    embedding_demo.train_word2vec(sentences)
    
    # Demonstrate word similarities
    test_words = ["king", "dog", "cute"]
    for word in test_words:
        if word in embedding_demo.model.wv:
            embedding_demo.find_similar_words(word, use_pretrained=False)
    
    # Try to load pre-trained model for better analogies
    print("\n" + "="*40)
    print("PRE-TRAINED MODEL DEMONSTRATION")
    print("="*40)
    
    has_pretrained = embedding_demo.load_pretrained_model()
    
    if has_pretrained:
        # Demonstrate famous analogies with pre-trained model
        print("\n🎯 Testing famous word analogies:")
        
        analogies = [
            ("man", "king", "woman"),  # man:king :: woman:?
            ("Paris", "France", "Berlin"),  # Paris:France :: Berlin:?
        ]
        
        for word1, word2, word3 in analogies:
            try:
                embedding_demo.word_analogy(word1, word2, word3, use_pretrained=True)
            except:
                print(f"❌ Could not compute analogy for {word1}:{word2}::{word3}:?")
    
    # Show vector representation
    print("\n📊 Vector Representation Example:")
    if "dog" in embedding_demo.model.wv:
        vector = embedding_demo.get_word_vector("dog")
        print(f"  Vector for 'dog' (first 10 dimensions): {vector[:10]}")
        print(f"  Vector size: {len(vector)}")


def interactive_demo():
    """Interactive demo for users to try"""
    print("\n\n" + "=" * 60)
    print("🎮 INTERACTIVE DEMO")
    print("=" * 60)
    
    print("Try the TF-IDF calculator with your own documents!")
    print("Enter 'quit' to exit")
    
    documents = []
    while True:
        doc = input(f"\nEnter document {len(documents) + 1} (or 'done' to finish): ")
        if doc.lower() == 'quit':
            return
        elif doc.lower() == 'done':
            break
        elif doc.strip():
            documents.append(doc)
    
    if not documents:
        print("No documents entered. Skipping interactive demo.")
        return
    
    # Calculate TF-IDF for user documents
    tfidf_calc = TFIDFCalculator()
    tfidf_calc.add_documents(documents)
    
    print(f"\n🔤 Vocabulary: {sorted(tfidf_calc.vocabulary)}")
    
    while True:
        term = input("\nEnter a term to check its TF-IDF score (or 'quit'): ")
        if term.lower() == 'quit':
            break
        
        if term in tfidf_calc.vocabulary:
            for i in range(len(documents)):
                tfidf_calc.calculate_tfidf(i, term)
        else:
            print(f"❌ Term '{term}' not found in vocabulary")


def main():
    """Main function to run all demonstrations"""
    print("🎓 WELCOME TO THE TEXT REPRESENTATION LEARNING APP!")
    print("This app teaches you how to convert text into numerical vectors.")
    print("\nWe'll cover two main methods:")
    print("1. 🔢 TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("2. 🧠 Word Embeddings (Word2Vec)")
    
    # Run demonstrations
    demonstrate_tfidf()
    demonstrate_word_embeddings()
    
    # Interactive section
    while True:
        choice = input("\n🎮 Would you like to try the interactive demo? (y/n): ").lower()
        if choice in ['y', 'yes']:
            interactive_demo()
            break
        elif choice in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("\n🎉 Thank you for learning about text representation!")
    print("📖 Check out lesson.md for detailed theory and homework.md for practice!")


if __name__ == "__main__":
    main() 