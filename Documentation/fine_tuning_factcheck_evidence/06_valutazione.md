# Step 6 — Valutazione: verdetto invariato, evidenza corretta, costo invariato

## Obiettivo

Verificare, per ciascuno dei tre modelli, tre cose:
1. il verdetto non è peggiorato rispetto al modello del piano binario;
2. le frasi evidenziate sono quelle giuste, anche quando sono lontane fra loro o lontane dal claim;
3. la latenza resta quella di un decision model.

## 1. La suite

| Blocco | Cosa misura | Quando |
|---|---|---|
| dev_select, test_id, test_ood, AggreFact | verdetto (come nel piano binario) | dev sempre; test una volta |
| dev_evid_select / confirm | evidenza: scelte e conferma | sempre / una volta per fase |
| dev_evid_neg | frasi sopra soglia sui negativi | sempre |
| test_evid_id (FEVER, HoVer, C2D) | evidenza in distribuzione | una volta |
| test_evid_ood (SciFact, WiCE) | evidenza fuori dominio | una volta |
| test_evid_sparse | evidenza lontana | una volta |
| Latenza | 1.000 richieste reali di lunghezza mista | modello finale, per modello |

## 2. Metriche di evidenza

Calcolate sulle righe **supportate e giudicate supportate** dal modello (solo lì si evidenzia):

| Metrica | Definizione | Perché |
|---|---|---|
| Precisione / recall / F1 per frase | sulle frasi mostrate | qualità di base |
| **Recall dell'insieme completo** | quota di righe in cui **tutte** le frasi gold sono mostrate | un'evidenza multi-hop a metà non convince l'utente |
| Match esatto dell'insieme | frasi mostrate = frasi gold | molto severa; utile per confronti |
| AUPRC per frase | indipendente dalla soglia (punteggio `s_k − s_0`) | ordinamento delle frasi |
| Errore sul numero di frasi | \|mostrate − gold\| medio | dice se la soglia dell'istanza fa il suo lavoro |
| F1 per fascia di distanza fra frasi gold | adiacenti / 2–9 / 10–19 / ≥ 20 frasi | evidenza sparsa |
| **F1 per distanza in token dal preambolo** | < 1.024 / 1.024–4.096 / > 4.096 (ultima frase gold) | il claim arriva alle frasi lontane? ([00c](00c_modelli_e_architetture.md) §3) |
| **Sufficienza** | le sole frasi mostrate, come documento, danno P(supportato) sopra soglia | non dipende dalle annotazioni incomplete |
| Necessità | togliendo le frasi mostrate, P scende sotto soglia | indica se l'evidenza mostrata è davvero quella usata (con documenti ridondanti può fallire anche se l'evidenza è giusta: riportare, non vincolare) |

Sufficienza e necessità costano 2 forward per riga, nel formato sandwich (il preambolo resta): si calcolano solo in valutazione, mai in produzione. Si calcolano con il modello stesso; per JPT e Winnow, se le etichette per ablazione vengono da d1, si riporta anche la sufficienza misurata con d1.

Per FEVER con più insiemi gold: una riga conta come "insieme completo" se le frasi mostrate contengono **almeno uno** degli insiemi.

Su `dev_evid_neg`: quota di righe con almeno una frase sopra `s_0`, e quota fra quelle con una frase lessicalmente simile al claim.

## 3. Metriche del verdetto

Identiche al piano binario ([../fine_tuning_factcheck/06_valutazione.md](../fine_tuning_factcheck/06_valutazione.md)), confrontate con il modello del piano binario **nello stesso formato**. Con le fasi 1 (backbone congelato) devono essere identiche: è un controllo che pipeline e rendering non siano cambiati.

## 4. Latenza

| Misura | Come | Obiettivo |
|---|---|---|
| Latenza p50/p95 | stesso server, stesse richieste, con e senza testa | ≤ +3% (d1: ≤ +3% o ≤ 2 ms) |
| Costo del preambolo | stesso server, con e senza preambolo, senza testa | riportare a parte: è un costo del formato, non della testa |
| Throughput | richieste/s con batching | ≤ −3% |
| Overhead della testa | aggregatore + pooling per frase + riga di attenzione negli 8 layer | < 1 ms per richiesta |
| Trasferimento degli stati | solo n vettori per layer del mix dalla GPU | trascurabile; se il server restituisce tutti gli stati, non lo è: va evitato ([08](08_merge_export.md)) |

## 5. Comportamentali

| Test | Atteso |
|---|---|
| Evidenza spostata (inizio/metà/fine) | stesse frasi evidenziate, nuove posizioni |
| Evidenza spostata oltre 1.024 / 4.096 token | stesse frasi evidenziate |
| Distrattori aggiunti | nessun distrattore evidenziato |
| Frase di evidenza duplicata altrove nel documento | entrambe evidenziate o almeno una; mai nessuna |
| Claim supportato da una frase che non condivide parole (parafrasi) | evidenziata comunque (E-LEX fallisce qui) |
| Coppia minima C2D (positivo / negativo con una frase tolta) | sul positivo le frasi giuste; sul negativo verdetto "no" e nessuna frase sopra `s_0` |

## 6. Report finale (formato, una colonna di modello per blocco)

| Blocco | Metrica | E-LEX | E-ATT | E-ABL (offline) | Testa, fase 1 (d1 / JPT / Winnow) | Testa, fase 2 |
|---|---|---:|---:|---:|---:|---:|
| test_evid_id | F1 / insieme completo | | | | | |
| test_evid_ood | F1 / insieme completo | | | | | |
| test_evid_sparse | F1 ≥ 20 frasi; F1 > 4.096 token | | | | | |
| tutti | sufficienza | | | | | |
| verdetto | BAcc media dei blocchi | — | — | — | = piano binario | |
| latenza | p50 / p95 | | | n·k forward | | |

## 7. Codice

Da scrivere: `finetune/eval_evidence.py` (metriche di §2, fasce di distanza in frasi e in token, sufficienza e necessità), estensione dei benchmark per il formato sandwich, `finetune/latency.py`.
