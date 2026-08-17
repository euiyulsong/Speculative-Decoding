import re
import string
from collections import Counter

def make_prompt(x, demo):
    gold = demo["answers"]["text"][0]
    return "\n".join([
        "Answer using only a short answer from the context.",
        "",
        "Example",
        f"Context: {demo['context']}",
        f"Question: {demo['question']}",
        f"Answer: {gold}",
        "",
        "Now answer the following.",
        f"Context: {x['context']}",
        f"Question: {x['question']}",
        "Answer:",
    ])

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punc(text):
        return "".join(c for c in text if c not in string.punctuation)
    return " ".join(remove_articles(remove_punc(s.lower())).split())

def exact_match(pred, golds):
    pred = normalize_answer(pred)
    return max(float(pred == normalize_answer(g)) for g in golds)

def f1_score(pred, gold):
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)

def best_f1(pred, golds):
    return max(f1_score(pred, g) for g in golds)
