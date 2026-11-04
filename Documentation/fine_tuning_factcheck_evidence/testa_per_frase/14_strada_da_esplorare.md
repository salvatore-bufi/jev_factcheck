# Che cosa non è implementato, e da dove partire

L'implementazione copre la pipeline **obbligatoria** del piano per **KnowLine-4B** (decisione del 2026-11-04). Per ogni parte mancante: che cosa servirebbe e quale codice esistente si può riusare.

| Parte del piano | Stato | Che cosa servirebbe | Riuso |
|---------------------------------------------------------|---------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **Altri decision model** (elephant-1, Jeffan, d3-flash, OneJev, Nimble, StartLux, pplx, Clef) | no | per ogni modello: un renderer con il suo prompt (stato in JSON → la mappa `escape_with_map` della doc del piano 06 §3) e, per i bidirezionali, la loro maschera dentro `sdpa_capture` (`CAPTURE["inner"]`) | `render.token_char_spans` e `sentence_token_spans` sono già generici; `model.SentenceTap` e `attention_per_sentence` funzionano su qualunque Qwen3.5 |
| **Dati generati** (passo 03) | no | generatore LLM + giudice; ablazione; evidenza spostata | `baselines.ablation_group`, `extract.rebuild`, ricetta del mix ([03](03_dati_generati.md)) |
| **`test_evid_sparse`** e prove di comportamento | no | evidenza spostata nei test | `extract.rebuild` |
| **Fase 1b** (H2d: media pesata dei token) | no | training con il modello in memoria e i vettori di tutti i token della frase | `SentenceTap` andrebbe esteso per restituire i token |
| **Fase 2** (LoRA del modello + testa) | no | gradiente attraverso il modello; controllo di BAcc ed ECE a ogni checkpoint | `forward_sentences` senza `no_grad` |
| **E-EMB** | no | `sentence-transformers` + `BAAI/bge-small-en-v1.5` | `baselines.evaluate_method` |
| **Sonda per layer** (doc del piano 15 §3) | no | estrazione su tutti i 32 layer per circa 700 righe, poi una regressione logistica per layer | aggiungere layer a `config.SENTENCE_LAYERS` e usare `extract.py` |
| **Importanza controfattuale** (doc del piano 14) | no | un forward per frase tolta | `extract.rebuild`, `forward_sentences(features=False)` |
| **Verdetto fine-tunato** (LoRA del fine-tuning binario) | non esiste nella repo | caricare la LoRA in `model.load_model`; eventualmente cambiare `config.QUESTION` | il resto della pipeline non cambia |

## La strada più promettente

La **sonda per layer** costa poco e risponde a una domanda aperta: i layer 19, 23, 27 e 31 sono davvero i migliori? Si ottiene già, in parte, dai pesi $\pi$ della miscela dopo H2a ([08](08_testa_e_loss.md) §2.A).

Il secondo passo naturale sono gli **altri modelli**, a partire da elephant-1, che è bidirezionale nei layer pieni. Lì l'esperimento H4 (preambolo sì o no) diventa interessante, perché le frasi vedono il claim anche senza preambolo.
