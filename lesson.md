# From Words to Vectors 📝➡️🔢

Machine learning models, like those used in AI, understand numbers, not words. To analyze, classify, or generate text, we must first convert it into a numerical format. This process is called **text representation**. The goal is to create numerical vectors that represent words, sentences, or entire documents.

---

## Method 1: TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents (a "corpus"). It's a great way to give more weight to important words and less weight to common ones. It's calculated by multiplying two metrics:

### **Term Frequency (TF)**

This measures how often a word appears in a single document. A higher value means the word is more important *within that document*.

* **Formula:** $TF(t, d) = \frac{\text{Number of times term 't' appears in document 'd'}}{\text{Total number of terms in document 'd'}}$

### **Inverse Document Frequency (IDF)**

This measures how common or rare a word is across all documents. It penalizes common words (like "the", "a", "is") that appear everywhere.

* **Formula:** $IDF(t, D) = \log\left(\frac{\text{Total number of documents 'D'}}{\text{Number of documents containing term 't'}}\right)$

### **Putting It Together (TF-IDF)**

The final TF-IDF score is the product of TF and IDF. A high score means the word is frequent in a specific document but rare across the entire collection.

* **Final Score:** $TF-IDF(t, d, D) = TF(t, d) \times IDF(t, D)$

**Simple Example:**

Imagine we have two documents:
* **Doc 1:** "The dog is cute."
* **Doc 2:** "My dog is a good boy."

Let's find the TF-IDF score for the word **"cute"** in **Doc 1**.

1.  **TF:** "cute" appears 1 time in Doc 1, which has 4 words.
    * $TF = 1 / 4 = 0.25$
2.  **IDF:** The word "cute" appears in only 1 of the 2 total documents.
    * $IDF = \log(2 / 1) = 0.301$
3.  **TF-IDF Score:**
    * $TF-IDF(\text{"cute", Doc 1}) = 0.25 \times 0.301 = 0.075$

Now consider the word **"is"** in **Doc 1**. It appears in both documents, so its IDF is $\log(2/2) = \log(1) = 0$. This makes its final TF-IDF score **0**, correctly identifying it as an unimportant word.

---

## Method 2: Word Embeddings (e.g., Word2Vec, GloVe)

Word Embeddings are a modern approach where words are represented as dense, low-dimensional vectors. Unlike TF-IDF, embeddings capture a word's **semantic meaning** and its relationships with other words. 🧠

### **How it Works**

Models like **Word2Vec** are trained on massive text corpora. They learn that **words appearing in similar contexts have similar meanings**.

This allows us to do "word math" that reflects these relationships. The vector for a word captures its essence.

**Simple Example:**

The relationships between words are encoded mathematically. This allows for amazing analogies:

* `vector('king') - vector('man') + vector('woman')` results in a vector that is very close to `vector('queen')`.

* Similarly, for capitals and countries:
    `vector('Paris') - vector('France') + vector('Germany')` results in a vector very close to `vector('Berlin')`.

### **Key Characteristics**

* **Dense Vectors:** Each word is a vector of a fixed size (e.g., 300 numbers), not thousands.
* **Contextual Meaning:** Embeddings capture semantic relationships, synonyms, and analogies.
* **Pre-trained Models:** You can download embeddings (from Google, etc.) that have already learned from the entire internet, saving you significant training time.

---

## TF-IDF vs. Word Embeddings: A Quick Comparison

| Feature | TF-IDF | Word Embeddings |
| :--- | :--- | :--- |
| **Core Idea** | Word frequency and rarity | Semantic context and meaning |
| **Vector Type** | Sparse (many zeros) | Dense (meaningful numbers) |
| **Context** | No, treats words in isolation | Yes, captures relationships |
| **Best For** | Keyword extraction, document search | Semantic search, sentiment analysis | 