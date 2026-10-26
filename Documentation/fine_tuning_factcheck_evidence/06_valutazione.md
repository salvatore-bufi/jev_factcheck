# Step 6 — Valutazione: verdetto invariato, evidenza corretta, costo invariato

## Obiettivo

Verificare tre cose:
1. il verdetto non è peggiorato rispetto al modello del piano binario;
2. le frasi evidenziate sono quelle giuste, anche quando sono lontane fra loro;
3. la latenza resta quella di un decision model.

## 1. La suite

| Blocco | Cosa misura | Quando |
|---|---|---|
| dev_select, test_id, test_ood, AggreFact | verdetto (come nel piano binario) | dev sempre; test una volta |
| dev_evid_select / confirm | evidenza: scelte e conferma | sempre / una volta per fase |
| test_evid_id (FEVER, HoVer, C2D) | evidenza in distribuzione | una volta |
| test_evid_ood (SciFact, WiCE) | evidenza fuori dominio | una volta |
| test_evid_sparse | evidenza lontana | una volta |
| Latenza | 1.000 richieste reali di lunghezza mista | modello finale |

## 2. Metriche di evidenza

Calcolate sulle righe **supportate e giudicate supportate** dal modello (solo lì si evidenzia):

| Metrica | Definizione | Perché |
|---|---|---|
| Precisione / recall / F1 per frase | sulle frasi mostrate (sopra `τ_e`) | qualità di base |
| **Recall dell'insieme completo** | quota di righe in cui **tutte** le frasi gold sono mostrate | un'evidenza multi-hop a metà non convince l'utente |
| Match esatto dell'insieme | frasi mostrate = frasi gold | molto severa; utile per confronti |
| AUPRC per frase | indipendente da `τ_e` | ordinamento delle frasi |
| F1 per fascia di distanza | adiacenti / 2–9 / 10–19 / ≥ 20 frasi | evidenza sparsa |
| **Sufficienza** | le sole frasi mostrate, come documento, danno P(supportato) sopra soglia | non dipende dalle annotazioni incomplete |
| Necessità | togliendo le frasi mostrate, P scende sotto soglia | indica se l'evidenza mostrata è davvero quella usata (con documenti ridondanti può fallire anche se l'evidenza è giusta: riportare, non vincolare) |

Sufficienza e necessità costano 2 forward per riga: si calcolano solo in valutazione, mai in produzione.

Per FEVER con più insiemi gold: una riga conta come "insieme completo" se le frasi mostrate contengono **almeno uno** degli insiemi.

## 3. Metriche del verdetto

Identiche al piano binario ([../fine_tuning_factcheck/06_valutazione.md](../fine_tuning_factcheck/06_valutazione.md)), confrontate con il modello del piano binario **con lo stesso ordine** claim → documento. Con la fase 1 (backbone congelato) devono essere identiche: è un controllo che pipeline e rendering non siano cambiati.

## 4. Latenza

| Misura | Come | Obiettivo |
|---|---|---|
| Latenza p50/p95 | stesso server, stesse richieste, con e senza testa | ≤ +3% |
| Throughput | richieste/s con batching | ≤ −3% |
| Overhead della testa | tempo dell'aggregatore da solo | < 1 ms per richiesta |
| Trasferimento degli stati | n × d valori dalla GPU (solo posizioni di fine frase) | trascurabile; se il server restituisce tutti gli stati, non lo è: va evitato ([08](08_merge_export.md)) |

## 5. Comportamentali

| Test | Atteso |
|---|---|
| Evidenza spostata (inizio/metà/fine) | stesse frasi evidenziate, nuove posizioni |
| Distrattori aggiunti | nessun distrattore evidenziato |
| Frase di evidenza duplicata altrove nel documento | entrambe evidenziate o almeno una; mai nessuna |
| Claim supportato da una frase che non condivide parole (parafrasi) | evidenziata comunque (E-LEX fallisce qui) |

## 6. Report finale (formato)

| Blocco | Metrica | E-LEX | E-ATT | E-ABL (offline) | Testa, fase 1 | Testa, fase 2 |
|---|---|---:|---:|---:|---:|---:|
| test_evid_id | F1 / insieme completo | | | | | |
| test_evid_ood | F1 / insieme completo | | | | | |
| test_evid_sparse | F1 ≥ 20 frasi | | | | | |
| tutti | sufficienza | | | | | |
| verdetto | BAcc media dei blocchi | — | — | — | = piano binario | |
| latenza | p50 / p95 | | | n·k forward | | |

## 7. Codice

Da scrivere: `finetune/eval_evidence.py` (metriche di §2, fasce di distanza, sufficienza e necessità con vLLM), estensione dei benchmark per l'ordine claim → documento, `finetune/latency.py`.
