import re
from gensim.models import Word2Vec
from typing import List, Iterable, Tuple

#Minimal corpus (50 sentences), self contained
CORPUS = [
    "The king sits on his ancient throne.",
    "The queen loves her people dearly.",
    "A man walks through the city streets.",
    "A woman reads a book in the cafe.",
    "Paris is the capital of France.",
    "Berlin is a vibrant city in Germany.",
    "London has many historic landmarks.",
    "The happy child smiled at the dog.",
    "Using a computer helps solve problems quickly.",
    "Fast cars race on the long highway.",
    "The queen and king host a grand ball.",
    "A programmer writes code on the computer.",
    "The speedy runner finished the race fast.",
    "A dog is a loyal animal.",
    "Cats are often independent creatures.",
    "A woman bought fresh bread from the market.",
    "A man climbed the tall mountain.",
    "The city of Paris attracts many tourists.",
    "Berlin has a famous art scene.",
    "London's museums house historic treasures.",
    "The happy family went on a picnic.",
    "Computer science is a fascinating field.",
    "People enjoy fast internet connections.",
    "The queen wore a magnificent crown.",
    "The king addressed the crowd with pride.",
    "A man enjoys running every morning.",
    "A woman practices yoga daily.",
    "Paris hosts many cultural events.",
    "Berlin's nightlife is quite popular.",
    "London's weather can be unpredictable.",
    "A computer can process data quickly.",
    "Fast trains connect major European cities.",
    "The happy student received a good grade.",
    "The queen listened to her advisors.",
    "The king decided to lower taxes.",
    "A man plays guitar in the park.",
    "A woman paints beautiful landscapes.",
    "Paris offers delicious pastries.",
    "Berlin features modern architecture.",
    "London celebrates diverse cultures.",
    "Computer games entertain many people.",
    "Fast food is available on every corner.",
    "The happy couple planned their wedding.",
    "The king traveled to a distant land.",
    "The queen donated to charity.",
    "A man studied hard for his exams.",
    "A woman wrote a novel about love.",
    "Paris is famous for the Eiffel Tower.",
    "Berlin hosts international conferences.",
    "London holds one of the oldest parliaments.",
]


def preprocess(corpus: Iterable[str]) -> List[List[str]]:
    #Lowercase and tokenize each sentence.
    sentences = []
    for sentence in corpus:
        tokens = re.sub(r"[^\w\s]", "", sentence.lower()).split()
        if tokens:
            sentences.append(tokens)
    return sentences


def train_model(sentences: List[List[str]], *, vector_size: int, window: int, min_count: int) -> Word2Vec:
    #Train a Word2Vec model with the provided parameters.
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
    )


def show_analogies(model: Word2Vec, analogies: Iterable[Tuple[str, str, str]]) -> None:
    #Display analogy results for given tuples (a, b, c).
    for a, b, c in analogies:
        try:
            result = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)[0]
            print(f"{a} : {b} :: {c} : {result[0]} ({result[1]:.4f})")
        except KeyError as exc:
            print(f"Word not in vocabulary: {exc}")


def show_clusters(model: Word2Vec, words: Iterable[str]) -> None:
    #Print top five words similar to each word in *words*.
    for word in words:
        if word not in model.wv:
            print(f"'{word}' not in vocabulary")
            continue
        similar = model.wv.most_similar(word, topn=5)
        formatted = ", ".join(f"{w}({s:.2f})" for w, s in similar)
        print(f"Similar to {word}: {formatted}")


def main() -> None:
    sentences = preprocess(CORPUS)

    #Try a few different hyperparameter settings
    configs = [
        {"vector_size": 50, "window": 3, "min_count": 1},
        {"vector_size": 100, "window": 5, "min_count": 1},
        {"vector_size": 50, "window": 5, "min_count": 2},
    ]

    analogies = [
        ("king", "man", "woman"),
        ("paris", "france", "berlin"),
        ("walk", "walking", "run"),
    ]

    for cfg in configs:
        print("\n=== Training model", cfg, "===")
        model = train_model(sentences, **cfg)
        show_analogies(model, analogies)
        show_clusters(model, ["happy", "computer", "fast"])


if __name__ == "__main__":
    main()