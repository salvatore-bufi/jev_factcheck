# Step 0 — Costruire e congelare gli split

## Obiettivo

Prima di qualsiasi misura e prima di generare qualunque dato, produrre e **congelare** questi insiemi:

| Insieme | A cosa serve | Quando si guarda |
|---|---|---|
| **train** | l'unica fonte di training e di generazione | sempre |
| **dev_select** | ogni scelta: soglia, temperatura, checkpoint, esperimenti | sempre |
| **dev_confirm** | conferma delle scelte fatte su dev_select | una volta per fase |
| **test_id** | test delle fonti usate in training | una volta per la baseline, una alla fine |
| **test_ood** | fonti mai usate (generalizzazione) | idem |
| **test_long** | righe di test_id e test_ood con documenti lunghi | idem |
| **AggreFact test** | confronto con la leaderboard (zero-shot) | idem |

## Perché prima di tutto

1. **AggreFact zero-shot.** Fino a oggi soglie e temperature si sono fittate sul dev di AggreFact. Se tutto si sceglie sul nostro dev, il risultato su AggreFact è confrontabile con la leaderboard.
2. **Stesso protocollo per baseline e modello fine-tunato.**
3. **Niente fughe.** La generazione ([02b](02b_dataset_generazione_llm.md)) parte solo dal train: se gli split non fossero fissati prima, un documento di test potrebbe finire nel training in forma modificata.

## 1. Regole per fonte

Regola generale: test ufficiali dove sono etichettati; altrimenti divisione **per documento** (`doc_id`).

| Fonte | train | dev (poi select/confirm) | test_id | Note |
|---|---|---|---|---|
| FEVER | train ufficiale | `paper_dev` | `paper_test` | `build_pairs.py fever --splits train dev test` |
| ANLI | train R1–R3 | dev R1–R3 | test R1–R3 | `--splits train dev test` |
| VitaminC | train | validation | test | `--splits train dev test` |
| HoVer | train | metà del dev ufficiale (per doc_id) | l'altra metà | il test ufficiale non ha etichette |
| AVeriTeC | train | metà del dev ufficiale | l'altra metà | idem |
| MiniCheck C2D/D2C | 80% dei doc_id | 10% | 10% | un solo split all'origine |
| SummEdits | 80% | 10% | 10% | i riassunti dello stesso documento restano insieme |

Fonti **solo test_ood** (mai in train né dev):

| Fonte | Caricamento | Documento | Claim | Etichetta 1 se |
|---|---|---|---|---|
| SciFact | `allenai/scifact_entailment`, `revision="refs/convert/parquet"`, split train + validation | `Title: <title>\n` + frasi di `abstract` unite da spazio | `claim` | `verdict == "SUPPORT"` |
| Climate-FEVER | `tdiggelm/climate_fever`, split test | per ognuna delle 5 `evidences`: `Title: <article>\n<evidence>`, unite da riga vuota | `claim` | `claim_label == 0` (0 = SUPPORTS, 1 = REFUTES, 2 = NOT_ENOUGH_INFO); scartare 3 = DISPUTED |
| HealthVer | `dwadden/healthver_entailment`, `revision="refs/convert/parquet"` (il repo ha uno script di caricamento non più supportato) | come SciFact | `claim` | `verdict == "SUPPORT"` (verificare i nomi delle colonne con `ds.features`) |
| COVID-Fact | `dwadden/covidfact_entailment`, idem | come SciFact | `claim` | idem |
| HaluBench | `PatronusAI/HaluBench`, split test | `Question: <question>\n\n<passage>` | `answer` | `label == "PASS"`; le righe con `source_ds` RAGTruth si riportano **a parte** (RAGTruth è anche in AggreFact) |
| HalluMix | `quotientai/HalluMix` | `Question: <question>\n\n` + i `documents` uniti da riga vuota | `answer` | `hallucination_label` falso (verificare il tipo: nel dataset è una stringa "True"/"False") |
| LLM-Oasis | `Babelscape/LLM-Oasis_claim_verification`, split `gold` | `evidence` | `claim` | campo etichetta da verificare con `ds.features` (nel primo esempio letto compaiono solo `id`, `claim`, `evidence`) |

Esclusi del tutto: **RAGTruth** (98% dei documenti in comune con AggreFact), **WiCE** (fonte di AggreFact), **LLM-AggreFact** dev e test dal training (vietato dalla licenza), **FactKG** in forma di triple.

## 2. Dimensioni

| Insieme | Righe per fonte | Motivo |
|---|---|---|
| dev_select | fino a 600, bilanciate 50/50 | errore della BAcc per fonte di ~±3 punti |
| dev_confirm | fino a 400, bilanciate | conferma |
| test_id | fino a 2.000, bilanciate | errore di ~±2 punti per fonte |
| test_ood | tutto (non si bilancia: si usa la BAcc) | |

Con 7 fonti: dev ≈ 7.000 righe, test_id ≈ 12.000.

## 3. Regole contro le fughe

1. **doc_id disgiunti**: nessun `doc_id` in più di uno split. Le righe di train con un doc_id presente in dev o test si tolgono dal train.
2. **Quasi-duplicati**: due documenti diversi possono essere quasi identici (pagine Wikipedia in FEVER e VitaminC). Si toglie dal train ogni riga il cui documento condivide ≥ 20% dei 13-grammi campionati con un documento di dev o test.
3. **Decontaminazione esterna**: dal train si tolgono le righe che si sovrappongono (claim identico normalizzato, o ≥ 20% dei 13-grammi) con LLM-AggreFact (dev + test) e con tutte le fonti di test_ood.
4. **Generazione solo dal train** (Step 2b); le righe generate ereditano il `doc_id` della riga di partenza.
5. **Dev e test con etichette originali**: niente dati generati in dev e test.

### Come funziona il controllo a 13-grammi

```text
parole = re.findall(r"\w+", testo.lower())
per ogni finestra di 13 parole consecutive: h = blake2b(" ".join(finestra), 8 byte) come intero
si tiene h solo se h % 8 == 0         (campionamento deterministico: 1/8 dei 13-grammi)
sovrapposizione(doc) = |H(doc) ∩ H(riferimento)| / |H(doc)|
scarto se sovrapposizione ≥ 0,20
```

Esempio: un documento di 400 parole ha 388 finestre, ~48 campionate. Se 12 di queste compaiono fra i 13-grammi dei documenti di test, sovrapposizione = 0,25 → scartato. È lo stesso codice di `shingles()` in [data_prep/make_mix.py](../../data_prep/make_mix.py).

## 4. Procedura

### 4.1 Produrre le coppie con tutti gli split

```sh
conda run -n factcheck python data_prep/build_pairs.py fever anli vitaminc --splits train dev test
conda run -n factcheck python data_prep/build_pairs.py hover averitec --splits train dev
conda run -n factcheck python data_prep/build_pairs.py minicheck summedits scifact
```

(FEVER legge 7 GB di pagine Wikipedia: alcuni minuti. I download HF a volte falliscono con un errore 500 del proxy: rilanciare.)

### 4.2 Lo script `data_prep/make_splits.py` (da scrivere)

Codice di riferimento, completo nella logica:

```python
"""Freeze train / dev_select / dev_confirm / test_id / test_ood splits.  Run once."""
import hashlib, json, random, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/finetune/raw", ROOT / "data/finetune/splits"
SEED = 13
CAPS = {"dev_select": 600, "dev_confirm": 400, "test_id": 2000}

def bucket(key, mod=10_000):                      # posizione deterministica 0..9999 di un doc_id
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % mod

def assign(row, source):
    """-> 'train' | 'dev' | 'test_id' secondo le regole di §1."""
    s, b = row["split"], bucket(row["doc_id"])
    if source in ("fever", "anli", "vitaminc"):
        return {"train": "train", "dev": "dev", "test": "test_id"}[s]
    if source in ("hover", "averitec"):
        if s == "train":
            return "train"
        return "dev" if b < 5000 else "test_id"
    if source in ("minicheck", "summedits"):
        return "train" if b < 8000 else ("dev" if b < 9000 else "test_id")
    raise ValueError(source)

def balanced(rows, cap, rng):
    rng.shuffle(rows)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    k = min(len(pos), len(neg), cap // 2)
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out

def main(sources=("fever", "anli", "vitaminc", "hover", "averitec", "minicheck", "summedits")):
    rng = random.Random(SEED)
    from make_mix import load_aggrefact, shingles, norm_claim   # stessa decontaminazione del mix
    agg_sh, agg_claims = load_aggrefact()
    ood_sh, ood_claims = load_test_ood_index()     # stessi indici, sulle fonti di test_ood (§5)
    manifest = {"seed": SEED, "files": {}, "dropped": {}}
    for src in sources:
        parts = defaultdict(list)
        for line in open(RAW / f"{src}.jsonl"):
            r = json.loads(line)
            parts[assign(r, src)].append(r)
        held_docs = {r["doc_id"] for k in ("dev", "test_id") for r in parts[k]}
        held_sh = set().union(*(shingles(r["document"]) for k in ("dev", "test_id") for r in parts[k]))
        drop = defaultdict(int)
        train = []
        for r in parts["train"]:
            sh = shingles(r["document"])
            frac = lambda ref: len(sh & ref) / len(sh) if sh else 0.0
            if r["doc_id"] in held_docs:                       drop["doc_in_dev_test"] += 1
            elif frac(held_sh) >= 0.2:                          drop["near_dup_dev_test"] += 1
            elif norm_claim(r["claim"]) in agg_claims | ood_claims: drop["claim_in_external_test"] += 1
            elif frac(agg_sh) >= 0.2:                           drop["doc_in_aggrefact"] += 1
            elif frac(ood_sh) >= 0.2:                           drop["doc_in_test_ood"] += 1
            else:                                               train.append(r)
        dev = parts["dev"]
        select = [r for r in dev if bucket("sel:" + r["doc_id"]) < 6000]
        confirm = [r for r in dev if bucket("sel:" + r["doc_id"]) >= 6000]
        out = {"train": train,
               "dev_select": balanced(select, CAPS["dev_select"], rng),
               "dev_confirm": balanced(confirm, CAPS["dev_confirm"], rng),
               "test_id": balanced(parts["test_id"], CAPS["test_id"], rng)}
        for split, rows in out.items():
            path = OUT / split / f"{src}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            manifest["files"][f"{split}/{src}.jsonl"] = {
                "rows": len(rows), "pos": sum(r["label"] for r in rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["dropped"][src] = dict(drop)
    # test_ood: scritti da build_test_ood() (§5), poi aggiunti al manifest con lo stesso schema
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

Note:

- per HoVer e AVeriTeC il bucket separa il dev ufficiale in due metà (dev e test_id);
- `bucket("sel:" + doc_id)` usa una chiave diversa per non correlare la divisione select/confirm con quella dev/test;
- la quota di righe scartate per motivo va letta: se una fonte perde > 5% per quasi-duplicati, controllare a mano 20 casi.

### 4.3 Verifiche automatiche dopo il congelamento

```python
ids = defaultdict(set)
for split in ("train", "dev_select", "dev_confirm", "test_id", "test_ood"):
    for f in (OUT / split).glob("*.jsonl"):
        for line in open(f):
            ids[split].add(json.loads(line)["doc_id"])
for a, b in itertools.combinations(ids, 2):
    assert not (ids[a] & ids[b]), (a, b, len(ids[a] & ids[b]))
```

## 5. Le fonti di test_ood

Una funzione per fonte, con lo stesso schema di riga di `build_pairs.py` (`id, source, split="test", document, claim, label, orig_label, doc_id, meta`). Esempio per Climate-FEVER:

```python
def build_climate_fever():
    from datasets import load_dataset
    ds = load_dataset("tdiggelm/climate_fever", split="test")
    names = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}
    for ex in ds:
        lab = int(ex["claim_label"])
        if lab == 3:                       # DISPUTED: ambiguo per un'etichetta binaria
            continue
        doc = "\n\n".join(f"Title: {e['article']}\n{e['evidence']}" for e in ex["evidences"])
        yield row("climate_fever", "test", ex["claim_id"], doc, ex["claim"], lab == 0, names[lab])
```

Per HaluBench:

```python
def build_halubench():
    ds = load_dataset("PatronusAI/HaluBench", split="test")
    for ex in ds:
        doc = f"Question: {ex['question']}\n\n{ex['passage']}"
        yield row("halubench", "test", ex["id"], doc, ex["answer"], ex["label"] == "PASS", ex["label"],
                  subset=ex["source_ds"])
```

Prima di scrivere ogni builder: `print(ds.features)` e due esempi, per confermare i nomi dei campi indicati in §1.

Per la decontaminazione del train (§3, regola 3) serve l'indice dei 13-grammi e dei claim di test_ood: `load_test_ood_index()` legge i file di `splits/test_ood/` e restituisce `(set di shingles dei documenti, set di claim normalizzati)`, con le stesse funzioni di `make_mix.py`.

## 6. test_long

Insieme di righe di test_id e test_ood il cui **prompt renderizzato** supera 4.000 token con il tokenizer di JPT. Si salvano solo gli id:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("kirp/jpt-9b", revision="b447cc7ee105c0a76a22f8fde8ecf05074dc8be0")
long_ids = [r["id"] for r in rows if len(tok(r["document"], add_special_tokens=False)["input_ids"]) > 3800]
(OUT / "test_long/ids.json").write_text(json.dumps(long_ids))
```

(3.800 token di documento + ~200 di prompt ≈ 4.000.) Se test_long ha meno di 300 righe, aggiungere documenti lunghi di HalluMix o abbassare la soglia a 3.000 token, e dichiararlo.

## 7. Come si aggregano i risultati

1. BAcc per fonte.
2. Media non pesata per blocco: test_id, test_ood, AggreFact (media sulle 11 fonti, come la leaderboard).
3. **Numero unico** = media dei tre blocchi.
4. AggreFact riportato anche da solo, nel formato della leaderboard.

Esempio (numeri inventati): test_id 88,5; test_ood 77,0; AggreFact 78,6 → media 81,4.

## 8. Le soglie

- **Principale**: temperatura e soglia fittate su `dev_select`, confermate su `dev_confirm`, applicate a tutti i test.
- **Soglia fissa 0,5**: riportata sempre; se la BAcc a 0,5 è entro 0,5 punti da quella con soglia fittata, il modello è ben calibrato.
- **Secondaria, dichiarata**: fittata sul dev di AggreFact (righe di `dev_split.py`), solo per confrontarsi con i numeri già prodotti in questa repo.

Il dev di AggreFact non entra in nessun'altra decisione.

## 9. Output

```text
data/finetune/splits/
  train/<fonte>.jsonl
  dev_select/<fonte>.jsonl
  dev_confirm/<fonte>.jsonl
  test_id/<fonte>.jsonl
  test_ood/<fonte>.jsonl
  test_long/ids.json
  manifest.json
```

Dopo il congelamento i file di dev e test **non si modificano più**. Ogni risultato riporta lo SHA-256 di `manifest.json`.

## Criteri per passare allo step successivo

- `manifest.json` scritto; la verifica §4.3 passa; scarti per motivo registrati.
- test_ood costruito per almeno SciFact, Climate-FEVER, HaluBench.
- test_long con almeno 300 righe.
