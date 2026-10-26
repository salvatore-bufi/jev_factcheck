# Le opzioni di paper, a confronto

La domanda è: *creare un dataset nuovo e testarci i modelli? proporre un dataset? oppure altro?* Rispetto all'8 ottobre le opzioni sono le stesse, ma il loro valore è cambiato: la valutazione di Jev su AggreFact è già pubblicata ([2609.37647](https://arxiv.org/abs/2609.37647)), e le ricette di fine-tuning dei TDM pure ([2610.02076](https://arxiv.org/abs/2610.02076), [2610.17205](https://arxiv.org/abs/2610.17205), [2609.35865](https://arxiv.org/abs/2609.35865)). Dettagli in [01](01_stato_arte_e_novita.md) e [09](09_aggiornamento_ottobre_2026.md).

**Vincolo valido per tutte le opzioni**: solo modelli fino a 9B parametri (taglia nominale). I modelli più grandi (Jev, i 27B della leaderboard, Winnow-12B e Rune-26B già misurati) servono solo come riferimento.

## Opzione 1 — Benchmarking ampio ("tutti i TDM su AggreFact")

**Cosa**: valutare molti TDM aperti e chiusi sul fact checking documento–claim (LLM-AggreFact + fuori dominio), contro checker specializzati e LLM generici.

| Pro | Contro |
|---|---|
| quasi tutta l'inferenza è già fatta o pronta | **più debole di prima**: Jev su AggreFact è pubblicato, la leaderboard ha già i TDM |
| tempi brevi | senza una domanda scientifica è "una tabella più lunga della leaderboard" |

**Adatta a**: workshop o preprint, non a una conferenza principale.

## Opzione 2 — Paper di analisi con domande scientifiche nette (**consigliata ora**)

**Cosa**: la stessa inferenza dell'Opzione 1, organizzata attorno a tre domande che nessuno ha affrontato sul grounding:
1. **Base contro ricetta** (RQ2): sette TDM su Qwen3.5-9B e tre su Qwen3.5-4B (Kev sulle varianti Base), contro la loro base letta in modo ingenuo e calibrato senza training. *Quanto aggiunge l'addestramento da decision model, e quanto varia fra ricette sulla stessa base?*
2. **Protocollo contro modello** (RQ3): quanto spostano il risultato soglia, polarità e parafrasi della domanda, una contro tre domande, composizione della richiesta, troncamento, contaminazione. *Le differenze della leaderboard sono più grandi del rumore di protocollo?*
3. **Che errori fanno** (RQ4–RQ6): ammissione di claim non supportati, contraddetto contro non stabilito, calibrazione e trasferimento della soglia, sovrapposizione degli errori.

Più **un solo esperimento di fine-tuning** (B-ctrl, [04](04_parte_B_finetuning.md)): stesso training di grounding su un TDM e sulla sua base.

| Pro | Contro |
|---|---|
| ogni risultato ha valore, in qualunque direzione esca ("la ricetta conta poco" è un risultato forte quanto "conta molto") | servono le basi generiche e i test di robustezza: ~3 settimane di lavoro |
| usa il vantaggio unico della repo: stesse domande, cache di logit, grid search senza inferenza, misure di costo ed energia | il fine-tuning ridotto a un controllo: chi cerca un metodo resta deluso |
| risponde alla domanda aperta dell'audit (vantaggio della lettura tipizzata) sul compito dove conta | il campo corre: i risultati sui modelli vanno congelati a una data |
| sta nei tempi di ARR del 4 gennaio | il vincolo dei 9B esclude i TDM migliori della leaderboard (27B): il paper non può dire "il miglior fact checker", ma "il miglior fact checker che si può far girare ovunque" |
| con il vincolo dei 9B tutto gira su una GPU da 24–48 GB: niente rischio di calcolo | |

**Adatta a**: ACL 2027 (main o Findings).

## Opzione 3 — Valutazione + metodo di specializzazione (era la consigliata)

**Cosa**: l'Opzione 2 più il fine-tuning completo (loss sulle lettere, replay o ancoraggio, varianti di domanda, negativi difficili, ablazioni, tre seed) su JPT o elephant-1 (9B) e su un 4B.

| Pro | Contro |
|---|---|
| contributo "dove sono i TDM" + "come migliorarli" | **la novità del metodo è calata**: loss sulle lettere + KL (LLM-as-Jev), calo di calibrazione con il fine-tuning (Inner Jev), coppie contrastive e rimozione dell'evidenza (PACT) sono già pubblicati in forma generale |
| | 2–2,5 mesi di lavoro: non entra nel 4 gennaio insieme alla Parte A fatta bene |
| | rischio che il fine-tuning non superi i migliori fino a 9B (Nimble-v4 77,8, Bespoke-MiniCheck-7B 77,4); i 27B (79,0–79,5) sono un tetto difficile |

**Adatta a**: secondo paper (EMNLP 2027 o ciclo ARR successivo), meglio se combinato con l'Opzione 4 completa e con dati multilingue.

## Opzione 4 — Set diagnostico ricavato da test esistenti

**Cosa**: test controllati costruiti trasformando esempi di test esistenti (coppie minime, rimozione dell'evidenza, distrattori, posizione, polarità della domanda, composizione della richiesta), validati a mano su un campione. Dettagli: [05_set_diagnostico.md](05_set_diagnostico.md).

| Pro | Contro |
|---|---|
| misura proprio le proprietà che i lavori di ottobre mettono in dubbio (polarità, contesto che ribalta, richieste composte) | coppie minime controllate esistono già per domini singoli (radiologia, finanza): il valore sta nel farle su molti TDM e su fonti diverse |
| economico | non si può derivare da LLM-AggreFact (licenza CC BY-ND, solo valutazione) |

Nel primo paper: **versione ridotta** (le famiglie automatiche che non richiedono validazione umana, più rimozione dell'evidenza dove è annotata). La versione completa va nel secondo.

## Opzione 5 — Proporre un dataset nuovo

| Pro | Contro |
|---|---|
| contributo chiaro e durevole | costoso (settimane di annotazione), revisori severi, allontana dalla domanda principale |
| con **etichette distribuzionali** (più annotatori per esempio) sarebbe l'unico modo per valutare davvero la calibrazione, promessa centrale dei TDM; Sys1Cal ([2609.35342](https://arxiv.org/abs/2609.35342)) e ProbabilityBench ([2610.17016](https://arxiv.org/abs/2610.17016)) lo fanno su dati sintetici, non sul grounding | |

**Quando**: secondo o terzo paper (NeurIPS 2027 Datasets & Benchmarks, LREC). Candidati distintivi: etichette distribuzionali per il grounding, con la classe "non stabilito" separata da "contraddetto"; documenti in italiano.

## Tabella riassuntiva

| Opzione | Novità oggi | Costo | Tempo | Rischio di rifiuto (main) | Venue tipica |
|---|---|---|---|---|---|
| 1. Benchmarking ampio | bassa | basso | 3–4 sett. | alto | workshop, preprint |
| **2. Analisi con domande nette + B-ctrl** | **media-alta** | basso-medio | 7–8 sett. | medio | **ACL 2027 main/Findings** |
| 3. Analisi + metodo completo | media (metodo in parte pubblicato) | medio | 3–4 mesi | medio | EMNLP 2027 |
| 4. Set diagnostico | media (come aggiunta) | basso-medio | 2–3 sett. | — | dentro la 2 (ridotto) o la 3 (completo) |
| 5. Dataset nuovo | alta se il buco è reale | alto | 4–6 mesi | medio-alto | D&B, LREC |

## Raccomandazione

**Opzione 2, con il set diagnostico ridotto e B-ctrl, sottomessa ad ARR il 4 gennaio 2027 per ACL 2027, con un preprint entro fine novembre.** Poi l'Opzione 3 (+ Opzione 4 completa) come secondo paper.

Motivi:
- la valutazione "piatta" e il metodo generale sono stati occupati da altri in ottobre; le domande RQ2 e RQ3 no, e la repo ha già quasi tutti i dati per rispondere;
- ogni esito è pubblicabile: è la proprietà che serve con dieci settimane a disposizione;
- B-ctrl è l'unico esperimento di fine-tuning con una domanda nuova (inizializzazione da TDM contro base, per il grounding) e costa poche corse;
- il preprint anticipato è indispensabile: tre lavori vicini in sei settimane.

## Cosa decidere subito

1. **Data di congelamento della lista dei modelli** (proposta: 30 novembre). Dopo quella data, nuovi modelli solo in appendice.
2. **I riferimenti fuori vincolo**: Jev 1.13 (per continuità con [2609.37647](https://arxiv.org/abs/2609.37647) e con il nostro run) e 1.14, che costano pochi dollari; Winnow-12B e Rune-26B già misurati; i 27B della leaderboard solo citati. Proposta: una riga "riferimenti" in fondo alla tabella principale, nessuna analisi su di loro.
3. **B-ctrl su quale coppia**: proposta JPT-9B (o elephant-1) contro Qwen3.5-9B, perché la base è la stessa di sette TDM; in alternativa KnowLine-4B o pplx-decider-flash contro Qwen3.5-4B, che costa la metà; d1-3B per il pilota.
