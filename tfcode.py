from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

# Assume cleaned_documents is imported or created
# For demonstration using sample documents
cleaned_documents = [
    ['showers','continued','throughout','week','bahia','cocoa','zone','alleviating','drought','since','early','january']
]

print("\n--- TERM FREQUENCY KEYWORDS ---\n")

doc_tokens = cleaned_documents[0]

word_freq = Counter(doc_tokens)

top_keywords = word_freq.most_common(10)

print("Top 10 Keywords (TF method):\n")

for word, freq in top_keywords:
    print(word, ":", freq)

print("\n--- TF-IDF KEYWORDS ---\n")

processed_texts = [" ".join(doc) for doc in cleaned_documents]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_texts)

feature_names = vectorizer.get_feature_names_out()

tfidf_scores = tfidf_matrix[0].toarray()[0]

word_scores = list(zip(feature_names, tfidf_scores))

sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

print("Top 10 Keywords (TF-IDF method):\n")

for word, score in sorted_words[:10]:
    print(word, ":", round(score, 4))

# Save keywords to CSV
with open("tfidf_keywords.csv","w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Keyword","Score"])

    for word,score in sorted_words[:10]:
        writer.writerow([word,score])

print("\nKeywords saved to tfidf_keywords.csv")
