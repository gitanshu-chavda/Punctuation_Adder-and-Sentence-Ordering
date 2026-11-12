import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"
FT = "models/punctuation/finetuned-quick"
TEXTS = [
    "भगवान राम के पूर्वज महाराज शिवि ने अपना वानप्रस्थ गढ़ मुक्तेश्वर में ही व्यतीत किया था चेंगलपट स्टेशन से अरकोनम जाने वाली लाइन पर कांजीवरम स्टेशन है",
    "स्टेशन का नाम कांजीवरम जरूर है किंतु नगर का नाम कांचीपुरम है",
    "यूँ तो हर तीर्थ बड़ा और अहम है लेकिन सात स्थानों की बड़ी महत्ता और मान्यता है",
]
ID2 = {0: "0", 1: ".", 2: ",", 3: "?", 4: "-", 5: ":"}


def punctuate(model_path, text):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForTokenClassification.from_pretrained(model_path)
    mdl.eval()
    words = text.split()
    enc = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = mdl(**{k: v for k, v in enc.items()}).logits[0]
    pred = logits.argmax(-1).tolist()
    wids = enc.word_ids()
    out = []
    seen = set()
    for i, w in enumerate(wids):
        if w is None or w in seen:
            continue
        seen.add(w)
        lab = ID2[pred[i]]
        if lab == ".":
            mark = "।"
        elif lab == "0":
            mark = ""
        else:
            mark = lab
        out.append(words[w] + mark)
    return " ".join(out)


for t in TEXTS:
    print("IN : " + t)
    print("BASE: " + punctuate(MODEL_ID, t))
    print("FT  : " + punctuate(FT, t))
    print("---")
