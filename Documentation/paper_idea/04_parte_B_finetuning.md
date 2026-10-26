# Parte B — Specializzare i decision model senza romperli

Il piano tecnico completo è in [../fine_tuning/](../fine_tuning/00_overview.md). Qui c'è solo ciò che serve **per il paper**: l'ipotesi, gli esperimenti indispensabili e come presentarli.

## 1. L'ipotesi

> Un typed decision model si può specializzare sul fact checking documento–claim con un fine-tuning leggero (LoRA) che ottimizza **solo la decisione che si legge in inferenza** (il softmax sui logit delle lettere delle opzioni), più un **replay con target del modello di partenza**. Si ottiene un guadagno netto sul fact checking, zero-shot fuori dominio, mantenendo calibrazione e capacità di decisione generali.

## 2. Contributi tecnici (cosa c'è di nuovo)

1. **Loss sulle lettere** al posto dell'SFT classico: si ottimizza solo il softmax ristretto alle opzioni nella posizione della risposta. Mostrare con dati che l'SFT classico peggiora la calibrazione (ECE) a parità di accuratezza.
2. **Replay per self-distillation** per un modello i cui dati originali non sono pubblici: domande tipizzate generiche, con target uguali alle probabilità del modello congelato (KL). Misurare l'oblio su JevBench, Kev e typed-decisions.
3. **Dati solo testo contro testo, con negativi difficili**: fonti pubbliche + negativi generati per cancellazione dell'evidenza e modifica minima del claim, con etichette derivate dall'operazione e verificate da un secondo modello.
4. **Robustezza del formato**: varianti automatiche di domanda e nomi dei campi, per non trasformare il TDM in un classificatore a una sola stringa.

Nessuno di questi elementi da solo è radicalmente nuovo. La novità è **combinarli per una classe di modelli nuova (i TDM)** e **misurare il compromesso specializzazione / oblio / calibrazione**, che l'audit del filone indica come problema aperto.

## 3. Esperimenti indispensabili per il paper

| ID | Esperimento | Perché un revisore lo chiederà |
|---|---|---|
| B-main | Winnow-FT e JPT-FT contro la baseline, su AggreFact zero-shot + fuori dominio + in distribuzione | il risultato principale |
| B-ctrl | **Stesso training su Gemma-4-12B e Qwen3.5-9B base** | separa il contributo dell'inizializzazione da decision model da quello dei dati |
| B-loss | Loss sulle lettere contro SFT classico; CE contro Brier | giustifica la scelta della loss |
| B-replay | Con e senza replay; quota di replay (0 / 10 / 20%) | misura l'oblio evitato |
| B-data | Leave-one-source-out (o almeno: solo MiniCheck+ANLI, + negativi generati, mix completo) | quali dati contano |
| B-var | Con e senza varianti di domanda | robustezza al wording |
| B-seed | 3 seed per la configurazione finale | varianza |
| B-ext | Bespoke-MiniCheck-7B e FactCG con la stessa suite | confronto con lo stato dell'arte aperto |

Facoltativi (se c'è tempo): loss contrastiva su coppie minime; S3 (tre domande) contro S1; rank LoRA; 8k contro 4k token.

## 4. Possibili esiti e come raccontarli

| Esito | Lettura | Come impostare il paper |
|---|---|---|
| FT del TDM ≫ baseline e ≫ FT della base generica, oblio contenuto | l'inizializzazione da decision model aiuta | paper di **metodo**: "Specializing typed decision models for grounded fact checking" |
| FT del TDM ≈ FT della base generica | il valore è nei dati; il TDM non dà vantaggio di partenza | paper di **analisi**: risultato onesto e utile ("decision training does not transfer to grounding") |
| FT migliora AggreFact ma peggiora fuori dominio | sovra-specializzazione | analisi delle fonti di dati (B-data); paper di analisi |
| Oblio forte anche con replay | compromesso reale | curva specializzazione contro oblio come risultato centrale |

Tutti gli esiti sono pubblicabili **se** il protocollo è pulito e i controlli ci sono. Il rischio vero è un paper senza B-ctrl o senza oblio misurato.

## 5. Calcolo e tempi (stime)

| Voce | Stima |
|---|---|
| Codice (render, train, merge, valutatori) | 2–3 settimane |
| Generazione di replay, varianti, negativi difficili (modello locale o API) | 1 settimana |
| Corse su JPT (pilota + ablazioni, ~10 corse) | 2–3 settimane su portatile o L40S |
| Corse su Winnow (~5 corse) + controlli sulle basi (2 corse) | 2 settimane su L40S |
| Valutazione completa ed export | 1 settimana |
| **Totale Parte B** | **~2–2,5 mesi** in parallelo con la scrittura |

## 6. Cosa rilasciare

- Adapter LoRA (o pesi fusi) di Winnow-FT (licenza da verificare in base ai dati, vedi [../fine_tuning/08_merge_export.md](../fine_tuning/08_merge_export.md)) e JPT-FT (non commerciale).
- Codice di training e valutazione.
- Ricette dei dati (script, non i dati di terze parti, se le licenze non lo consentono) e set diagnostico.
- Cache delle previsioni sui test, come già fatto per Jev in questa repo: permettono di ricalcolare le metriche senza nuova inferenza.
