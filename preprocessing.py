import os
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Path to extracted Reuters dataset
extract_path = r"C:\Users\dines\Downloads\reuters21578"

documents = []

# Read all SGML files
sgm_files = [f for f in os.listdir(extract_path) if f.endswith(".sgm")]

for file in sgm_files:
    file_path = os.path.join(extract_path, file)

    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()
        soup = BeautifulSoup(content, "html.parser")

        for reuter in soup.find_all("reuters"):
            body = reuter.find("body")
            if body is not None:
                documents.append(body.get_text().strip())

print("Total BODY documents extracted:", len(documents))

# Preprocessing
stop_words = set(stopwords.words('english'))

cleaned_documents = []

for doc in documents:
    doc = doc.lower()
    doc = re.sub(r'[^a-z\s]', '', doc)
    tokens = word_tokenize(doc)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    cleaned_documents.append(tokens)

print("\nPreprocessing completed!")

print("\nSample Cleaned Tokens:")
print(cleaned_documents[0][:20])
