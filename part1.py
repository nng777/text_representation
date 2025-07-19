
from typing import List, Tuple

from app import TFIDFCalculator


class ExtendedTFIDFCalculator(TFIDFCalculator):
    #TFIDFCalculator with extra methods.

    def find_most_important_words(self, doc_index: int, top_n: int = 5) -> List[Tuple[str, float]]:
        #Return top N words with highest TF-IDF scores for the document.
        if doc_index >= len(self.documents):
            return []
        scores = {}
        for term in self.vocabulary:
            scores[term] = self.calculate_tfidf(doc_index, term)
        # Sort by TF-IDF descending
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:top_n]

    def compare_documents(self, doc1_index: int, doc2_index: int, top_n: int = 5) -> List[str]:
        #Find common important words between two documents.
        words_doc1 = {word for word, _ in self.find_most_important_words(doc1_index, top_n)}
        words_doc2 = {word for word, _ in self.find_most_important_words(doc2_index, top_n)}
        return sorted(words_doc1 & words_doc2)


def task_1_1():
    #Manual TF-IDF calculations for provided movie reviews.
    corpus = [
        "This movie is absolutely fantastic and amazing",
        "The movie was terrible and boring",
        "Amazing acting but terrible plot",
        "Fantastic movie with great acting",
    ]

    calc = ExtendedTFIDFCalculator()
    calc.add_documents(corpus)

    #Calculate TF-IDF for specific terms with intermediate output
    calc.calculate_tfidf(0, "amazing")
    calc.calculate_tfidf(1, "terrible")


def task_1_2():
    #Demonstrate new methods with a custom sentences/corpus.
    documents = [
        "Cats are wonderful companions",
        "Dogs are loyal and friendly animals",
        "Birds can fly high in the sky",
        "Fish swim in the ocean",
        "Many people love their pets",
    ]

    calc = ExtendedTFIDFCalculator()
    calc.add_documents(documents)

    print("\nMost important words in Document 1:")
    for word, score in calc.find_most_important_words(0):
        print(f"  {word}: {score:.4f}")

    print("\nMost important words in Document 2:")
    for word, score in calc.find_most_important_words(1):
        print(f"  {word}: {score:.4f}")

    common = calc.compare_documents(0, 1)
    print(f"\nCommon important words between Doc 1 and Doc 2: {common}")


if __name__ == "__main__":
    print("Running Task 1.1: Manual TF-IDF Calculation")
    task_1_1()

    print("\n\nRunning Task 1.2: Extended TFIDFCalculator")
    task_1_2()