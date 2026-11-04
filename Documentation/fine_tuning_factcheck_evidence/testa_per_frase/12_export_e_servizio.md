# Passo 12 — Rilascio e server

## In breve

- [`evidence/release.py`](../../../evidence/release.py) crea una cartella `releases/<nome>/` con tutto ciò che serve al server, congelato con degli hash.
- [`evidence/serve.py`](../../../evidence/serve.py) fa, per ogni richiesta, **un forward** e restituisce verdetto ed evidenza. Usa lo stesso codice della valutazione: ciò che si è misurato è ciò che gira.

```sh
python -m evidence.release --run data/evidence/runs/knowline-4b/<id> --name knowline-4b-fc-ev-v1
python -m evidence.release --verify releases/knowline-4b-fc-ev-v1
python -m evidence.serve --run releases/knowline-4b-fc-ev-v1 --port 8090
curl -s localhost:8090/v1/evidence -d '{"document": "...", "claim": "..."}'
python -m evidence.serve --run releases/knowline-4b-fc-ev-v1 --demo
```

## 1. Una richiesta (`Service.answer`)

$$
\begin{aligned}
\text{documento} &\xrightarrow{\ \texttt{split}\ } \{[s_k, e_k)\}
\xrightarrow{\ \texttt{render}\ } (\texttt{input\_ids},\ [a_k, b_k])\\
&\xrightarrow{\ \texttt{forward\_sentences}\ } (\ell_A, \ell_B,\ q,\ h^{end},\ h^{mean},\ att)\\
&\xrightarrow{\ P > 0{,}5\ ?\ } \text{testa} \to (s, s_\star) \xrightarrow{\ \texttt{select}\ } \mathcal{S}
\end{aligned}
$$

1. Divisione in frasi con `evidence/sentences.py`, lo stesso divisore del training.
2. Prompt con il preambolo con cui la testa è stata addestrata (`feature_meta.preamble` della run) e posizioni delle frasi.
3. Un forward con gli hook (`to_cpu=False`: le feature restano sulla GPU).
4. Se $P \le 0{,}5$: risposta senza evidenza.
5. Altrimenti: la testa sui layer della configurazione (`layers_read`), poi la regola della run ($\delta$ o $\tau_e$), poi le posizioni nel **documento originale**.

```json
{"p_supported": 0.9987, "supported": true,
 "evidence": [{"sentence": 0, "start": 0, "end": 77, "score": 0.8333, "text": "Saint Paul School of Theology is ..."},
              {"sentence": 4, "start": 185, "end": 259, "score": 0.6102, "text": "Nancy R. Howell is ..."}]}
```

**Regole del servizio:**
- `start` / `end` si riferiscono al documento così come l'ha mandato il client;
- `score` vale $\sigma(s_k - \theta)$;
- un documento oltre 32.768 token (`config.MAX_TOKENS_SERVE`) dà l'errore `document_too_long` e non viene mai troncato; un documento vuoto dà `empty_document`;
- un lock serializza i forward sulla GPU, con batch 1;
- endpoint HTTP: `POST /v1/evidence`, `GET /health` (`make_handler`, `ThreadingHTTPServer`).

## 2. La cartella di rilascio (`release.build`)

| File | Contenuto |
|---|---|
| `head.safetensors`, `config.json`, `selection.json` | la testa, la sua configurazione (impostazioni delle feature comprese), la regola di selezione |
| `decision.json` | tutte le impostazioni: modello e revisione, domanda, preambolo, soglia, layer, feature, selezione, divisore, limite di token |
| `sentences/` | copia di `sentences.py` e dei parametri inglesi di NLTK `punkt_tab` |
| `frozen.json` | sha256 della testa, di `sentences.py`, `render.py`, `model.py`, `config.py`, di ogni file di `punkt_tab`, e dei manifest di split e mix; commit git |
| `report.md` | metriche della run (dev, valutazioni, latenza, se ci sono) |
| `MODEL_CARD.md` | che cosa fa e che cosa non fa |

I pesi di KnowLine non si copiano: sono quelli della revisione indicata in `decision.json`. Una release non si sovrascrive mai.

## 3. Perché il divisore va rilasciato con il modello

La testa lavora su **frasi**. Se in produzione il documento fosse diviso anche di poco diversamente, la testa vedrebbe unità diverse da quelle su cui è stata addestrata, e gli indici restituiti non corrisponderebbero più. Il risultato di `split` dipende dal nostro codice, dalla versione di NLTK e dai file `punkt_tab`. All'avvio da una release, `Service` chiama `release.verify`, che ricalcola gli hash sulla macchina e si ferma se qualcosa è diverso. Lo stesso vale per `render.py` e `model.py`: cambiarli cambierebbe le posizioni o le feature.

## 4. Controlli prima del rilascio

| Controllo | Deve dare | Misurato (prova, 2026-11-04) |
|---|---|---|
| Server = valutazione, stesse righe | stesse frasi; $\lvert\Delta P\rvert < 0{,}005$ | 60/60 righe, $\Delta P = 0$ |
| `release --verify` | hash uguali | sì |
| Latenza | C − B ≤ 3% o 2 ms | +1,8 ms (+0,5%) |
| Test (`test_evid_id`, `test_evid_ood`) | una volta sola, sulla release | da fare |

## 5. Codice

| Funzione / classe | File | Ruolo |
|---|---|---|
| `Service.__init__`, `Service.answer` | `serve.py` | caricamento (con verifica) e una richiesta |
| `make_handler`, `main` | `serve.py` | HTTP e `--demo` |
| `build`, `hashes`, `verify`, `punkt_dir`, `model_card` | `release.py` | cartella di rilascio e hash |
