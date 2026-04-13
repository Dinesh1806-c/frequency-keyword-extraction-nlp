# Frequency-Based Keyword Extraction on Reuters-21578

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-Keyword%20Extraction-green)
![Dataset](https://img.shields.io/badge/Dataset-Reuters--21578-orange)
![Libraries](https://img.shields.io/badge/Dependencies-None%20(Pure%20Python)-brightgreen)

> **Natural Language Processing Project**  
> **Student:** B. Dinesh | **Register No:** 23BCS027  
> **Dataset:** Reuters-21578 | **Language:** Python 3 (Standard Library Only)

---

## What This Project Does

This project automatically extracts the most important keywords from news articles using **5 different frequency-based algorithms**, all implemented from scratch in pure Python — no NLTK, no scikit-learn, no external libraries.

Given a news article like *"BAHIA COCOA REVIEW"*, the system automatically identifies:
- `cocoa`, `bahia`, `comissaria` (TF-IDF)
- `bahia cocoa review`, `total crop estimates` (RAKE)
- Statistically significant terms (Log-Likelihood)

---

## Dataset

**Reuters-21578** — one of the most widely used benchmark datasets in NLP research.

| Property | Value |
|---|---|
| Total Articles | 4,971 |
| Total Tokens | 327,939 |
| Vocabulary Size | 23,369 unique terms |
| Avg Tokens/Article | 65 |
| Domain | Financial & commodity news (1987) |
| Format | SGML (.sgm files) |

Download the dataset: https://www.daviddlewis.com/resources/testcollections/reuters21578/

---

## 5 Methods Implemented

| # | Method | Type | Extracts Phrases | Needs Corpus |
|---|---|---|---|---|
| 1 | **TF-IDF** | Statistical | No | Yes |
| 2 | **RAKE** | Linguistic | Yes | No |
| 3 | **Log-Likelihood Ratio (G2)** | Statistical | No | Yes |
| 4 | **Position-Weighted TF-IDF** | Structural | No | Yes |
| 5 | **Co-occurrence Graph Centrality** | Graph-based | No | No |

### Method Formulas

**TF-IDF:**
```
TF(t,d)  = count(t in d) / total words in d
IDF(t,D) = log((|D|+1) / (df(t)+1)) + 1   [Laplace smoothed]
TF-IDF   = TF x IDF
```

**RAKE:** Splits text on stopwords → scores candidate phrases by word co-occurrence degree

**Log-Likelihood (G2):**
```
G2 = 2 x sum(O x ln(O/E))
Where O = observed frequency, E = expected frequency
```

**Position-Weighted TF-IDF:**
```
score(word) = TF-IDF(word) x boost
boost = 3.0 if word in title, else 1.0
```

**Co-occurrence Graph (PageRank):**
```
score(i) = (1-d)/N + d x sum_j [w(j->i)/sum_k w(j->k)] x score(j)
d = 0.85, window = 4, iterations = 30
```

---

## Project Structure

```
NLP-Keyword-Extraction/
│
├── keyword_extraction.py    # Main pipeline — all 5 methods
├── results.json             # Auto-generated output report
├── README.md                # This file
└── requirements.txt         # No external dependencies needed
```

---

## How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/NLP-Keyword-Extraction.git
cd NLP-Keyword-Extraction
```

### Step 2 — Download the Reuters-21578 dataset
Download `reuters21578.tar.gz` from:
https://www.daviddlewis.com/resources/testcollections/reuters21578/

Place it at:
```
C:\Users\YOUR_USERNAME\Downloads\reuters21578.tar.gz
```

### Step 3 — Update the path in the script
Open `keyword_extraction.py` and update line ~350:
```python
tar_path = r"C:\Users\YOUR_USERNAME\Downloads\reuters21578.tar.gz"
data_dir = r"C:\Users\YOUR_USERNAME\Downloads\reuters21578"
```

### Step 4 — Run
```bash
python keyword_extraction.py
```

### Expected Output
```
======================================================================
  REUTERS-21578  --  FREQUENCY-BASED KEYWORD EXTRACTION PIPELINE
======================================================================

[1/7] Loading Reuters-21578 corpus from ... 
      Loaded 4971 articles from 5 SGM files.
[2/7] Tokenising & preprocessing ...
[3/7] Building TF-IDF scores ...
[4/7] Extracting keywords using 5 methods on 10 sample articles ...
[5/7] Detailed output for first 3 articles ...
[6/7] Evaluation -- Macro-Averaged Precision / Recall / F1
[7/7] Corpus Statistics
      [OK] Full JSON report saved -> results.json

  PIPELINE COMPLETE  (runs in ~2 seconds)
```

---

## Results

### Evaluation (Macro-Averaged over 10 articles)

| Method | Precision | Recall | F1-Score |
|---|---|---|---|
| **TF-IDF** | 0.0500 | 0.1583 | **0.0648** ✓ Best F1 |
| RAKE | 0.0404 | **0.2500** | 0.0639 ✓ Best Recall |
| Log-Likelihood | 0.0400 | 0.1500 | 0.0557 |
| Position TF-IDF | 0.0400 | 0.1333 | 0.0489 |
| Co-occurrence Graph | 0.0400 | 0.1417 | 0.0523 |

### Sample Output — Article: "BAHIA COCOA REVIEW" (Gold Topic: cocoa)

| Method | Extracted Keywords |
|---|---|
| TF-IDF | cocoa, bahia, comissaria, times, sept, york, bags |
| RAKE | bahia cocoa review, total bahia crop estimates around |
| Log-Likelihood | bahia, comissaria, cocoa, times, sept, aug, york |
| Position TF-IDF | cocoa, bahia, comissaria, times, sept, york, bags |
| Co-occurrence | cocoa, sales, york, bahia, bags, smith, comissaria |

### Top 20 Corpus Terms (after stopword removal)

| Rank | Term | Frequency |
|---|---|---|
| 1 | billion | 2,505 |
| 2 | inc | 2,053 |
| 3 | company | 2,000 |
| 4 | net | 1,921 |
| 5 | corp | 1,705 |
| ... | ... | ... |

---

## Key Innovations

1. **Pure Python from scratch** — every formula hand-coded, no library wrappers
2. **Unified comparative pipeline** — all 5 methods on same input, same evaluation
3. **Domain-aware stopwords** — Reuters-specific noise words removed (`dlrs`, `mln`, `pct`, `reuter`)
4. **Position-Weighted TF-IDF** — title words get 3x boost (news-domain innovation)
5. **PageRank on word graphs** — semantic centrality, not just raw frequency
6. **Fully automated pipeline** — auto-extract tar.gz, auto-detect SGM files, auto-save JSON

---

## Why F1 Scores Are Low (Important Note)

The F1 scores (0.05–0.06) are **expected and normal** for unsupervised keyword extraction on Reuters. The gold labels are coarse category labels (`grain`) while the system extracts specific terms (`wheat`, `sorghum`, `barley`) — semantically correct but not string-matching. Even state-of-the-art published systems achieve only F1 = 0.10–0.25 on Reuters coarse labels.

---

## References

| Paper | Link |
|---|---|
| Salton & Buckley (1988) — TF-IDF | [Semantic Scholar](https://www.semanticscholar.org/paper/Term-weighting-approaches-in-automatic-text-Salton-Buckley/f40d6f1937b05ec674a5e9ac5fb0e85a3c3a95e5) |
| Rose et al. (2010) — RAKE | [ResearchGate](https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents) |
| Dunning (1993) — Log-Likelihood | [ACL Anthology](https://aclanthology.org/J93-1003/) |
| Mihalcea & Tarau (2004) — TextRank | [ACL Anthology](https://aclanthology.org/W04-3252/) |
| Brin & Page (1998) — PageRank | [Stanford PDF](http://infolab.stanford.edu/pub/papers/google.pdf) |
| Lewis (1997) — Reuters-21578 | [Official Dataset Page](https://www.daviddlewis.com/resources/testcollections/reuters21578/) |

---

## License

This project is submitted as an academic NLP assignment.  
**Author:** B. Dinesh | **Register No:** 23BCS027  
Feel free to use for learning and reference purposes.
