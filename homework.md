# Homework Assignment: Text Representation Mastery 📝✍️

## Objective
Apply the concepts of TF-IDF and Word Embeddings to solve real-world text analysis problems. This assignment will test your understanding of both statistical and neural approaches to text vectorization.

---

## Part 1: TF-IDF Deep Dive 🔢

### Task 1.1: Manual TF-IDF Calculation (20 points)

Given the following corpus of movie reviews:

```
Document 1: "This movie is absolutely fantastic and amazing"
Document 2: "The movie was terrible and boring" 
Document 3: "Amazing acting but terrible plot"
Document 4: "Fantastic movie with great acting"
```

**Requirements:**
1. Calculate the TF-IDF score for the word "amazing" in Document 1
2. Calculate the TF-IDF score for the word "terrible" in Document 2
3. Show all intermediate steps (TF and IDF calculations)
4. Explain why "amazing" has a different TF-IDF score compared to common words like "movie"

**Deliverable:** 
- Step-by-step calculations with explanations
- Written analysis of results (2-3 sentences)

### Task 1.2: Custom TF-IDF Implementation (25 points)

Extend the `TFIDFCalculator` class from the lesson app:

1. **Add a method** `find_most_important_words(doc_index, top_n=5)` that returns the top N words with highest TF-IDF scores for a given document
2. **Add a method** `compare_documents(doc1_index, doc2_index)` that finds common important words between two documents
3. **Test your methods** with a corpus of at least 5 documents on a topic of your choice

**Deliverable:**
- Python code with the extended class
- Test results showing the methods working
- Brief explanation of what the results tell us about the documents

---

## Part 2: Word Embeddings Exploration 🧠

### Task 2.1: Word Relationships Analysis (20 points)

Using either the lesson app or your own implementation:

1. **Train a Word2Vec model** on a corpus of at least 50 sentences (you can use news articles, book excerpts, or any text source)
2. **Find analogies** similar to "king - man + woman = queen" using your trained model
3. **Analyze semantic clusters** by finding words similar to: "happy", "computer", "fast"

**Requirements:**
- Use different hyperparameters (vector_size, window, min_count) and compare results
- Try at least 3 different analogies
- Document which hyperparameters work best and why

**Deliverable:**
- Training code and results
- Analysis of how hyperparameters affect the quality of word relationships
- Screenshots or text output of your analogy experiments

### Task 2.2: Pre-trained vs Custom Models (15 points)

Compare the performance of your custom-trained Word2Vec model with a pre-trained model:

1. **Load a pre-trained model** (Google News vectors or any other available model)
2. **Test the same analogies** from Task 2.1 on both models
3. **Analyze the differences** in results and explain why they occur

**Deliverable:**
- Comparison table showing results from both models
- Written explanation (3-4 sentences) of why pre-trained models might perform differently

---

## Part 3: Practical Application 🎯

### Task 3.1: Document Classification Preparation (20 points)

Create a mini document classification system:

1. **Collect 20 documents** from 2 different categories (e.g., sports vs technology news, positive vs negative reviews)
2. **Convert each document** to both TF-IDF and Word2Vec representations
3. **Calculate similarity scores** between documents using cosine similarity
4. **Analyze which method** (TF-IDF vs Word2Vec) better separates the two categories

**Requirements:**
- Use real text data (not the examples from the lesson)
- Implement cosine similarity calculation
- Create a simple visualization or table showing the separation

**Deliverable:**
- Python script with your implementation
- Dataset used (or clear instructions on how to obtain it)
- Analysis report comparing TF-IDF vs Word2Vec effectiveness

---

## Bonus Challenge 🏆 (Extra 10 points)

### Advanced Text Processing

Implement one of the following advanced features:

**Option A: Text Preprocessing Pipeline**
- Create a comprehensive text preprocessing pipeline that handles:
  - Stop word removal
  - Stemming or lemmatization
  - N-gram generation
- Compare TF-IDF results with and without preprocessing

**Option B: Visualization**
- Create visualizations of word embeddings using techniques like t-SNE or PCA
- Show how similar words cluster together in 2D space

**Option C: Real-world Application**
- Build a simple search engine that uses TF-IDF to rank documents by relevance to a query
- Include at least 50 documents and test with multiple queries

---

## Submission Guidelines 📋

### What to Submit:
1. **Code files** (.py format) with clear comments
2. **Results document** (PDF or Markdown) with all calculations, analyses, and screenshots
3. **Dataset or data sources** used for experiments
4. **README file** explaining how to run your code

### Formatting Requirements:
- Use clear variable names and add comments to your code
- Include output examples in your results document
- Explain your reasoning for design choices
- Show intermediate steps for manual calculations

### Evaluation Criteria:
- **Correctness** (40%): Accurate calculations and working code
- **Understanding** (30%): Clear explanations and analysis
- **Implementation Quality** (20%): Clean, well-commented code
- **Creativity** (10%): Interesting datasets or additional insights

---

## Tips for Success 💡

1. **Start early** - Some tasks require downloading large models
2. **Test incrementally** - Make sure each part works before moving to the next
3. **Document everything** - Include screenshots of your results
4. **Use real data** - More interesting than toy examples
5. **Ask questions** - If something is unclear, seek clarification

### Common Pitfalls to Avoid:
- Don't forget to handle edge cases (empty documents, unknown words)
- Remember to normalize vectors when calculating similarity
- Be careful with logarithms in IDF calculation (avoid log(0))
- Check that your Word2Vec model actually learned meaningful relationships

---

## Resources 📚

### Helpful Documentation:
- [scikit-learn TF-IDF Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)

### Dataset Ideas:
- [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/)
- [Movie Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Reuters News Dataset](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection)
- News articles from RSS feeds
- Product reviews from e-commerce sites

**Due Date:** [To be specified by instructor]  
**Total Points:** 100 + 10 bonus points

Good luck with your text representation journey! 🚀 