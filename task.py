"""
Part 1: TF-IDF Deep Dive
Task 1.1: Manual TF-IDF Calculation
Given the following corpus of movie reviews:
Document 1: "This movie is absolutely fantastic and amazing"
Document 2: "The movie was terrible and boring"
Document 3: "Amazing acting but terrible plot"
Document 4: "Fantastic movie with great acting"

Requirements:
1.Calculate the TF-IDF score for the word "amazing" in Document 1
2.Calculate the TF-IDF score for the word "terrible" in Document 2
3.Show all intermediate steps (TF and IDF calculations)
4.Explain why "amazing" has a different TF-IDF score compared to common words like "movie"

Deliverable:
1.Step-by-step calculations with explanations
2.Written analysis of results (2-3 sentences)

Task 1.2: Custom TF-IDF Implementation
Extend the TFIDFCalculator class from the lesson app:
1.Add a method find_most_important_words(doc_index, top_n=5) that returns the top N words with highest TF-IDF scores for a given document
2.Add a method compare_documents(doc1_index, doc2_index) that finds common important words between two documents
3.Test your methods with a corpus of at least 5 documents on a topic of your choice

Deliverable:
1.Python code with the extended class
2.Test results showing the methods working
3.Brief explanation of what the results tell us about the documents
"""