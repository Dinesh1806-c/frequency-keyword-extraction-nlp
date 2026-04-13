"""
==============================================================================
  FREQUENCY-BASED KEYWORD EXTRACTION ON REUTERS-21578
  Advanced NLP Pipeline -- Pure Python (No External Dependencies)
==============================================================================
Techniques Implemented:
  1. TF-IDF  (Term Frequency - Inverse Document Frequency)
  2. RAKE    (Rapid Automatic Keyword Extraction)
  3. Log-Likelihood Ratio (corpus-level keyword scoring)
  4. Position-Weighted TF-IDF (title/body distinction)
  5. Co-occurrence Graph Centrality (TextRank-inspired)
  6. Evaluation: Precision, Recall, F1 vs gold topics
  7. Comparative analysis across all methods
==============================================================================
"""

import re
import os
import math
import json
import heapq
import string
import collections
from html import unescape


# -----------------------------------------------------------------------------
# 1.  STOP-WORDS  (hand-crafted; covers news domain)
# -----------------------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll i'm
i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's will with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
said reuters reuter dlrs mln bln pct cts shr loss profit year ago
its also will told said would could may might per one two three four five
six seven eight nine ten new said last first since still even amid
""".split())


# -----------------------------------------------------------------------------
# 2.  DATA LOADING  -- parse Reuters SGM files
# -----------------------------------------------------------------------------

def parse_reuters_sgm(path):
    """Parse a single .sgm file -> list of article dicts."""
    with open(path, "r", encoding="latin-1") as f:
        raw = f.read()

    raw = unescape(raw)
    articles = []
    for block in re.findall(r"<REUTERS[^>]*>(.*?)</REUTERS>", raw, re.DOTALL):
        art = {}
        title_m = re.search(r"<TITLE>(.*?)</TITLE>", block, re.DOTALL)
        body_m  = re.search(r"<BODY>(.*?)</BODY>",  block, re.DOTALL)
        topics_m = re.findall(r"<TOPICS>.*?</TOPICS>", block, re.DOTALL)

        art["title"]  = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
        art["body"]   = re.sub(r"\s+", " ", body_m.group(1)).strip()  if body_m  else ""
        art["topics"] = re.findall(r"<D>(.*?)</D>", topics_m[0]) if topics_m else []
        art["text"]   = art["title"] + " " + art["body"]

        if art["text"].strip():
            articles.append(art)
    return articles


def load_corpus(data_dir, max_files=5):
    """Load up to max_files SGM files."""
    sgm_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir) if f.endswith(".sgm")
    ])[:max_files]

    corpus = []
    for path in sgm_files:
        corpus.extend(parse_reuters_sgm(path))
    return corpus


# -----------------------------------------------------------------------------
# 3.  TEXT PREPROCESSING
# -----------------------------------------------------------------------------

def tokenize(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords & short tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z\s'-]", " ", text)
    tokens = text.split()
    tokens = [t.strip("'-") for t in tokens]
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return tokens


def tokenize_sentences(text):
    """Split text into sentences (for RAKE)."""
    sentences = re.split(r"[.!?;,\n]+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_keep_stops(text):
    """Tokenize but keep stopwords (needed for RAKE phrase boundary detection)."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


# -----------------------------------------------------------------------------
# 4.  METHOD 1: TF-IDF
# -----------------------------------------------------------------------------

def build_tfidf(corpus_tokens):
    """
    Compute TF-IDF for every term in every document.
    Returns: list of {term: tfidf_score} dicts, one per document.
    """
    N = len(corpus_tokens)
    # Document frequency
    df = collections.Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))

    tfidf_docs = []
    for tokens in corpus_tokens:
        tf = collections.Counter(tokens)
        total = len(tokens) or 1
        scores = {}
        for term, count in tf.items():
            tf_val  = count / total
            idf_val = math.log((N + 1) / (df[term] + 1)) + 1  # smoothed
            scores[term] = tf_val * idf_val
        tfidf_docs.append(scores)
    return tfidf_docs


def tfidf_keywords(doc_scores, top_n=10):
    """Return top-N keywords for a document by TF-IDF score."""
    return [term for term, _ in heapq.nlargest(top_n, doc_scores.items(), key=lambda x: x[1])]


# -----------------------------------------------------------------------------
# 5.  METHOD 2: RAKE (Rapid Automatic Keyword Extraction)
# -----------------------------------------------------------------------------

def rake_extract(text, top_n=10):
    """
    Classic RAKE algorithm:
      - Split on stop-words to get candidate phrases
      - Score each word: freq(w) = occurrences, degree(w) = co-occurrences in phrases
      - word_score = degree(w) / freq(w)
      - phrase_score = sum of word scores in phrase
    """
    tokens = tokenize_keep_stops(text)
    # Build candidate phrases (split on stopwords)
    phrases = []
    current = []
    for tok in tokens:
        if tok in STOPWORDS or not re.match(r"[a-z]+", tok):
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        phrases.append(current)

    # Word frequency and degree
    word_freq   = collections.Counter()
    word_degree = collections.Counter()
    for phrase in phrases:
        for word in phrase:
            word_freq[word]   += 1
            word_degree[word] += len(phrase) - 1  # co-occurrence with phrase members

    # Word score
    word_score = {w: (word_degree[w] + word_freq[w]) / word_freq[w]
                  for w in word_freq}

    # Phrase score
    phrase_scores = {}
    for phrase in phrases:
        score = sum(word_score.get(w, 0) for w in phrase)
        key   = " ".join(phrase)
        phrase_scores[key] = max(phrase_scores.get(key, 0), score)

    top = heapq.nlargest(top_n, phrase_scores.items(), key=lambda x: x[1])
    return [phrase for phrase, _ in top]


# -----------------------------------------------------------------------------
# 6.  METHOD 3: Log-Likelihood Ratio (corpus-level significance)
# -----------------------------------------------------------------------------

def log_likelihood_ratio(corpus_tokens, focus_tokens):
    """
    Compare term frequencies in a focus document vs the entire corpus.
    High LLR -> term is significantly more common in focus than expected.
    G2 = 2 * sum(O * ln(O/E))
    """
    corpus_flat = [t for doc in corpus_tokens for t in doc]
    corpus_freq = collections.Counter(corpus_flat)
    focus_freq  = collections.Counter(focus_tokens)

    total_corpus = len(corpus_flat)
    total_focus  = len(focus_tokens)
    scores = {}

    for term, O11 in focus_freq.items():
        O12 = corpus_freq[term] - O11          # in corpus but not focus
        O21 = total_focus - O11                # not term in focus
        O22 = total_corpus - corpus_freq[term] - O21  # not term in corpus

        N = O11 + O12 + O21 + O22 or 1
        E11 = (O11 + O12) * (O11 + O21) / N
        E12 = (O11 + O12) * (O12 + O22) / N

        def safe_log(o, e):
            return o * math.log(o / e) if o > 0 and e > 0 else 0

        g2 = 2 * (safe_log(O11, E11) + safe_log(O12, E12))
        scores[term] = g2

    return scores


# -----------------------------------------------------------------------------
# 7.  METHOD 4: Position-Weighted TF-IDF (Title Boost)
# -----------------------------------------------------------------------------

def position_weighted_tfidf(article, all_corpus_tokens, df, N, title_weight=3.0):
    """
    Words in the title get a multiplier bonus -- titles are highly
    informative in news articles.
    """
    title_tokens = tokenize(article["title"])
    body_tokens  = tokenize(article["body"])
    all_tokens   = title_tokens + body_tokens

    tf     = collections.Counter(all_tokens)
    total  = len(all_tokens) or 1
    scores = {}
    title_set = set(title_tokens)

    for term, count in tf.items():
        tf_val  = count / total
        idf_val = math.log((N + 1) / (df[term] + 1)) + 1
        boost   = title_weight if term in title_set else 1.0
        scores[term] = tf_val * idf_val * boost

    return scores


# -----------------------------------------------------------------------------
# 8.  METHOD 5: Co-occurrence Graph + Centrality (TextRank-inspired)
# -----------------------------------------------------------------------------

def cooccurrence_centrality(tokens, window=4, top_n=10, iterations=30, damping=0.85):
    """
    Build a co-occurrence graph within a sliding window, then run
    a simplified PageRank-like centrality scoring.
    Returns top-N keywords by centrality.
    """
    unique_words = list(set(tokens))
    word_idx     = {w: i for i, w in enumerate(unique_words)}
    n            = len(unique_words)
    if n == 0:
        return []

    # Build adjacency (edge weights = co-occurrence count)
    adj = collections.defaultdict(lambda: collections.defaultdict(float))
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            w1, w2 = tokens[i], tokens[j]
            if w1 != w2:
                adj[w1][w2] += 1.0
                adj[w2][w1] += 1.0

    # Normalize adjacency
    norm_adj = {}
    for w in unique_words:
        nbrs    = adj[w]
        total   = sum(nbrs.values()) or 1
        norm_adj[w] = {nb: cnt / total for nb, cnt in nbrs.items()}

    # PageRank iterations
    scores = {w: 1.0 / n for w in unique_words}
    for _ in range(iterations):
        new_scores = {}
        for w in unique_words:
            rank = (1 - damping) / n
            rank += damping * sum(
                norm_adj[nb].get(w, 0) * scores[nb]
                for nb in adj[w]
            )
            new_scores[w] = rank
        scores = new_scores

    return [w for w, _ in heapq.nlargest(top_n, scores.items(), key=lambda x: x[1])]


# -----------------------------------------------------------------------------
# 9.  EVALUATION -- Precision / Recall / F1 vs gold standard topics
# -----------------------------------------------------------------------------

def evaluate(predicted_keywords, gold_topics):
    """
    Match predicted single-word keywords against Reuters gold topics.
    For multi-word predictions we expand to constituent words.
    """
    pred_set = set()
    for kw in predicted_keywords:
        pred_set.update(kw.lower().split())

    gold_set = set(t.lower().replace("-", " ").replace(" ", "") for t in gold_topics)
    pred_flat = set("".join(w.split()) for w in pred_set)

    tp = len(pred_flat & gold_set)
    fp = len(pred_flat - gold_set)
    fn = len(gold_set - pred_flat)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)
    return {"precision": precision, "recall": recall, "f1": f1}


# -----------------------------------------------------------------------------
# 10.  MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  REUTERS-21578  --  FREQUENCY-BASED KEYWORD EXTRACTION PIPELINE")
    print("=" * 70)

    # -- Load data ------------------------------------------------------------
    import tarfile
    tar_path = r"C:\Users\dines\Downloads\reuters21578.tar.gz"
    data_dir = r"C:\Users\dines\Downloads\reuters21578"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    sgm_files_exist = any(f.endswith(".sgm") for f in os.listdir(data_dir))
    if not sgm_files_exist:
        print("\n[*] Extracting dataset from tar.gz ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print("    Done extracting.")

    print(f"\n[1/7] Loading Reuters-21578 corpus from {data_dir} ...")
    corpus = load_corpus(data_dir, max_files=5)
    print(f"      Loaded {len(corpus)} articles from 5 SGM files.")

    # -- Tokenise entire corpus -----------------------------------------------
    print("\n[2/7] Tokenising & preprocessing ...")
    corpus_tokens = [tokenize(art["text"]) for art in corpus]

    # -- Build DF for TF-IDF --------------------------------------------------
    N  = len(corpus)
    df = collections.Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))

    # -- Build global TF-IDF --------------------------------------------------
    print("\n[3/7] Building TF-IDF scores ...")
    tfidf_docs = build_tfidf(corpus_tokens)

    # -- Select sample articles for detailed demo -----------------------------
    # pick articles that have gold topics and reasonable length
    sample_arts  = [(i, art) for i, art in enumerate(corpus)
                    if art["topics"] and len(art["text"]) > 200][:10]

    print(f"\n[4/7] Extracting keywords using 5 methods on {len(sample_arts)} sample articles ...")

    results = []
    method_scores = {
        "TF-IDF":            {"precision": [], "recall": [], "f1": []},
        "RAKE":              {"precision": [], "recall": [], "f1": []},
        "LogLikelihood":     {"precision": [], "recall": [], "f1": []},
        "PositionTF-IDF":    {"precision": [], "recall": [], "f1": []},
        "CooccurrenceGraph": {"precision": [], "recall": [], "f1": []},
    }

    for doc_idx, art in sample_arts:
        tokens = corpus_tokens[doc_idx]
        if not tokens:
            continue

        # Method 1: TF-IDF
        m1_kws = tfidf_keywords(tfidf_docs[doc_idx], top_n=10)

        # Method 2: RAKE
        m2_kws = rake_extract(art["text"], top_n=10)

        # Method 3: Log-Likelihood
        llr_scores = log_likelihood_ratio(corpus_tokens, tokens)
        m3_kws = [t for t, _ in heapq.nlargest(10, llr_scores.items(), key=lambda x: x[1])]

        # Method 4: Position-Weighted TF-IDF
        pos_scores = position_weighted_tfidf(art, corpus_tokens, df, N)
        m4_kws = [t for t, _ in heapq.nlargest(10, pos_scores.items(), key=lambda x: x[1])]

        # Method 5: Co-occurrence Graph
        m5_kws = cooccurrence_centrality(tokens, top_n=10)

        gold = art["topics"]

        for method, kws in [
            ("TF-IDF",            m1_kws),
            ("RAKE",              m2_kws),
            ("LogLikelihood",     m3_kws),
            ("PositionTF-IDF",    m4_kws),
            ("CooccurrenceGraph", m5_kws),
        ]:
            ev = evaluate(kws, gold)
            for metric in ("precision", "recall", "f1"):
                method_scores[method][metric].append(ev[metric])

        results.append({
            "article_id": doc_idx,
            "title":      art["title"],
            "gold_topics": gold,
            "tfidf":       m1_kws,
            "rake":        m2_kws,
            "log_likelihood": m3_kws,
            "position_tfidf": m4_kws,
            "cooccurrence":   m5_kws,
        })

    # -- Print detailed results for first 3 articles --------------------------
    print("\n[5/7] Detailed output for first 3 articles:\n")
    print("-" * 70)
    for r in results[:3]:
        print(f"\n[Article #{r['article_id']}]: {r['title'][:70]}")
        print(f"   Gold Topics    : {r['gold_topics']}")
        print(f"   TF-IDF         : {r['tfidf'][:7]}")
        print(f"   RAKE           : {r['rake'][:5]}")
        print(f"   Log-Likelihood : {r['log_likelihood'][:7]}")
        print(f"   Position TF-IDF: {r['position_tfidf'][:7]}")
        print(f"   Co-occurrence  : {r['cooccurrence'][:7]}")
        print("-" * 70)

    # -- Aggregate evaluation -------------------------------------------------
    print("\n[6/7] Evaluation -- Macro-Averaged Precision / Recall / F1\n")
    print(f"{'Method':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 55)

    avg_results = {}
    for method, scores in method_scores.items():
        p = sum(scores["precision"]) / len(scores["precision"]) if scores["precision"] else 0
        r = sum(scores["recall"])    / len(scores["recall"])    if scores["recall"]    else 0
        f = sum(scores["f1"])        / len(scores["f1"])        if scores["f1"]        else 0
        print(f"{method:<22} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
        avg_results[method] = {"precision": p, "recall": r, "f1": f}

    # Find best method
    best_method = max(avg_results, key=lambda m: avg_results[m]["f1"])
    print(f"\n  >> Best Method by F1: {best_method} "
          f"(F1 = {avg_results[best_method]['f1']:.4f})")

    # -- Corpus-level statistics -----------------------------------------------
    print("\n[7/7] Corpus Statistics\n")
    all_tokens  = [t for toks in corpus_tokens for t in toks]
    vocab       = set(all_tokens)
    freq        = collections.Counter(all_tokens)
    top20       = freq.most_common(20)

    print(f"  Total articles     : {N}")
    print(f"  Total tokens       : {len(all_tokens):,}")
    print(f"  Vocabulary size    : {len(vocab):,}")
    print(f"  Avg tokens/article : {len(all_tokens)//N if N else 0}")
    print(f"\n  Top 20 corpus terms (after stopword removal):")
    for rank, (term, count) in enumerate(top20, 1):
        bar = "#" * min(40, count // 50)
        print(f"    {rank:>2}. {term:<20} {count:>5}  {bar}")

    # -- Save JSON report ------------------------------------------------------
    report = {
        "corpus_stats": {
            "total_articles":     N,
            "total_tokens":       len(all_tokens),
            "vocabulary_size":    len(vocab),
            "avg_tokens_per_doc": len(all_tokens) // N if N else 0,
        },
        "evaluation": avg_results,
        "best_method": best_method,
        "sample_results": results,
        "top_20_corpus_terms": top20,
    }

    out_path = os.path.join(data_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [OK] Full JSON report saved -> {out_path}")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
