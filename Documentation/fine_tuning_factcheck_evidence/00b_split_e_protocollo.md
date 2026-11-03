# Step 0 — Split e protocollo, con l'evidenza

## Obiettivo

Riusare gli split congelati del piano binario ([../fine_tuning_factcheck/00b_split_e_protocollo.md](../fine_tuning_factcheck/00b_split_e_protocollo.md)) e aggiungere i **sottoinsiemi con evidenza annotata a livello di frase**, su cui si misura e si sceglie tutto ciò che riguarda l'evidenziazione. Gli split sono gli stessi per d1-3B, JPT-9B e Winnow-12B.

## 1. I blocchi

| Blocco | Contenuto | Uso |
|---|---|---|
| dev_select / dev_confirm | come nel piano binario | verdetto: soglia, temperatura, checkpoint |
| **dev_evid_select** | righe **supportate** di `dev_select` con evidenza gold (FEVER, HoVer, C2D) | checkpoint della testa, esperimenti H, eventuale `τ_e` |
| **dev_evid_confirm** | idem da `dev_confirm` | conferma |
| **dev_evid_neg** | righe **non supportate** di `dev_select` con documento lungo (FEVER REFUTES / NOT ENOUGH INFO con la pagina intera, HoVer NOT_SUPPORTED) | controllo della loss sui negativi: quota di righe con almeno una frase sopra la soglia dell'istanza ([04](04_loss.md) §3) |
| test_id, test_ood, AggreFact | come nel piano binario | verdetto, una volta alla fine |
| **test_evid_id** | righe supportate di test_id con evidenza gold (FEVER `paper_test`, metà del dev di HoVer, 10% di C2D) | evidenza in distribuzione |
| **test_evid_ood** | SciFact (rationale annotate), WiCE (frasi di evidenza annotate) | evidenza fuori dominio |
| **test_evid_sparse** | da test_evid_id: documenti in cui le frasi gold distano ≥ 10 frasi, più versioni costruite con le frasi gold spostate agli estremi (§3) | evidenza sparsa |

L'evidenza si misura sulle righe **supportate**; `dev_evid_neg` serve solo a controllare che sui negativi la testa non "accenda" frasi (in produzione non verrebbero mostrate, ma è un sintomo di scorciatoia lessicale).

## 2. Fonti con evidenza a livello di frase

| Fonte | Evidenza gold | Split | Note |
|---|---|---|---|
| FEVER | insiemi di (pagina, riga); più insiemi alternativi per claim | train / paper_dev / paper_test | annotazioni **incomplete**: altre frasi possono supportare senza essere annotate ([04](04_loss.md) §4) |
| HoVer | `supporting_facts` (titolo, indice di frase), 2–4 pagine | train / dev diviso a metà | multi-hop: è il banco di prova dell'evidenza sparsa |
| MiniCheck C2D | le frasi `s_i,1`, `s_i,2` (solo se rigenerato con posizioni, [02b](02b_dataset_generazione_llm.md)) | 80/10/10 per documento | |
| SciFact | frasi di *rationale* dell'abstract | solo test (OOD) | dominio biomedico |
| WiCE | frasi di evidenza della pagina citata | solo test (OOD) | è anche una fonte di AggreFact: mai in training |

## 3. Il test di evidenza sparsa costruito

Da ogni riga di test_evid_id con almeno 2 frasi gold:
1. si prende il documento;
2. si sposta la prima frase gold all'inizio e l'ultima alla fine (le altre restano), oppure si inseriscono paragrafi irrilevanti da altri documenti dello **stesso split di test** fra le frasi gold;
3. si controlla che il testo resti coerente (le frasi spostate non devono dipendere da pronomi risolti dalla frase precedente; se lo fanno, la riga si scarta).

Il verdetto resta "supportato" e l'evidenza resta la stessa, con nuove posizioni. Si riportano i risultati per:
- **distanza fra le frasi gold**: adiacenti, 2–9 frasi, 10–19, ≥ 20;
- **distanza in token dell'ultima frase gold dal preambolo con il claim**: < 1.024, 1.024–4.096, > 4.096 (la soglia di 1.024 è la finestra dei layer locali di Winnow, [00c](00c_modelli_e_architetture.md) §3).

## 4. Regole contro le fughe

Quelle del piano binario, più:
- le etichette di evidenza generate o ricavate per ablazione ([02b](02b_dataset_generazione_llm.md)) si producono **solo sul train**;
- i paragrafi distrattori di test_evid_sparse vengono da documenti di test, mai di train;
- l'etichettatore per ablazione è il modello del verdetto: non vede mai righe di dev o test durante l'etichettatura.

## 5. Aggregazione

- Verdetto: come nel piano binario.
- Evidenza: metriche per fonte ([06](06_valutazione.md)), poi media non pesata per blocco (test_evid_id, test_evid_ood, test_evid_sparse).
- Le metriche di evidenza si calcolano sulle righe supportate **che il modello giudica supportate** (perché solo lì evidenzia), e si riporta a parte la quota di righe supportate giudicate non supportate (nessuna evidenza mostrata). Questa quota cambia fra d1, JPT e Winnow: i confronti fra modelli si fanno anche sull'intersezione delle righe giudicate supportate da tutti.

## 6. Output

```text
data/finetune/splits/                 # del piano binario
data/finetune/evidence/
  dev_evid_select/<fonte>.jsonl       # righe con "sentences" e "evidence"
  dev_evid_confirm/<fonte>.jsonl
  dev_evid_neg/<fonte>.jsonl
  test_evid_id/<fonte>.jsonl
  test_evid_ood/{scifact,wice}.jsonl
  test_evid_sparse/<fonte>.jsonl
  manifest.json
```

## Criteri per passare allo step successivo

- Sottoinsiemi con evidenza congelati, con numero di righe e distribuzione delle due distanze (fra frasi gold, dal preambolo in token per ciascun tokenizer).
- Nessun documento di test usato come distrattore nel train e viceversa.
