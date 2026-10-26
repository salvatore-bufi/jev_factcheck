# Il set diagnostico

Rivisto il 26 ottobre 2026: due versioni (ridotta per il primo paper, completa per il secondo) e tre famiglie nuove, suggerite dai lavori di ottobre sulla fragilità dei TDM.

## 1. Perché un set diagnostico e non un dataset nuovo

Un benchmark come LLM-AggreFact dice **quanto** un modello sbaglia, non **perché**. I TDM promettono di leggere il documento, di rispondere in modo stabile alla stessa domanda posta in modi diversi e di dare probabilità calibrate. In ottobre tre lavori hanno mostrato che queste promesse non sono scontate:
- la **polarità** della domanda ("ogni claim è supportato?" contro "contiene un claim non supportato?") inverte o annulla l'AUC di cinque TDM su otto su RAGTruth ([Arize](https://arize.com/blog/decision-model-benchmark/));
- **aggiungere altre domande** alla stessa richiesta cambia le risposte dei TDM aperti ([2610.14631](https://arxiv.org/abs/2610.14631));
- un **contesto naturale** aggiunto ribalta decisioni corrette e sicure ([JevOut, 2609.30243](https://arxiv.org/abs/2609.30243), con contesti ottimizzati in modo avversario).

Un set diagnostico misura queste proprietà sul grounding, su molti TDM, con esempi costruiti in modo controllato. Le coppie minime esistono già per domini singoli e solo per Jev (radiologia [2609.27607](https://arxiv.org/abs/2609.27607), citazioni finanziarie [2610.08675](https://arxiv.org/abs/2610.08675)): il nostro valore aggiunto è la copertura (molti modelli, più fonti) e il collegamento con RQ2 (le ricette sulla stessa base sono ugualmente fragili?).

## 2. Da dove si parte

Solo da **test** di dataset con licenza compatibile e mai usati in training:
- test in distribuzione delle nostre fonti (FEVER `paper_test`, VitaminC test, MiniCheck e SummEdits nella parte di test);
- fonti fuori dominio (SciFact, Climate-FEVER, HealthVer, COVID-Fact).

**Non** da LLM-AggreFact: la licenza è **CC BY-ND 4.0** e l'accesso è concesso "come benchmark di valutazione", quindi non si possono ridistribuire derivati. Su AggreFact si possono applicare le trasformazioni automatiche in locale e pubblicare solo gli script e i risultati aggregati.

**HoVer**: è fra i benchmark pubblici della Decision Index, quindi probabilmente visto dai TDM recenti; usarlo solo per le famiglie in cui conta la variazione fra originale e perturbato, e segnalarlo.

## 3. Le famiglie di test

| Famiglia | Trasformazione | Etichetta attesa | Costruzione | Validazione umana | Versione |
|---|---|---|---|---|---|
| **Polarità della domanda** (nuova) | stessa decisione chiesta in forma positiva e negata | complementare | automatica | no | **ridotta** |
| **Composizione della richiesta** (nuova) | la domanda da sola, poi dentro una richiesta con 2–3 altre domande (le nostre o domande non pertinenti) | invariata | automatica | no | **ridotta** |
| **Opzioni invertite o rinominate** | ordine A/B scambiato; `yes/no` → `true/false`, `supported/not supported` | invariata | automatica | no | **ridotta** |
| **Campi rinominati** | `document`/`claim` → `passage`/`assertion`, `context`/`answer` | invariata | automatica | no | **ridotta** |
| **Distrattori** | paragrafi irrilevanti dello stesso dominio prima e dopo | invariata | automatica | no | **ridotta** |
| **Posizione dell'evidenza** | la frase di supporto all'inizio, a metà o alla fine di un documento lungo | invariata | automatica | no | **ridotta** |
| **Solo claim** | documento vuoto o non pertinente | 0 | automatica | no | **ridotta** |
| **Rimozione dell'evidenza** | si cancellano le frasi gold che supportano il claim | da 1 a 0 (non stabilito) | automatica dove l'evidenza è annotata (FEVER); con LLM altrove | sì | ridotta (solo FEVER), completa |
| **Contesto naturale compatibile** (nuova) | una frase di contesto plausibile che non cambia la risposta (non ottimizzata in modo avversario, a differenza di JevOut) | invariata | LLM + filtro | a campione | completa |
| **Coppia minima sul claim** | si cambia un numero, una data, un'entità o un'attribuzione | da 1 a 0 (contraddetto) | LLM + verifica automatica | sì | completa |
| **Coppia minima sul documento** | si cambia nel documento il valore citato dal claim | da 1 a 0 | VitaminC (già contrastivo) + LLM | a campione | completa |
| **Formulazione della domanda** | 10 parafrasi mai viste | invariata | LLM + filtro | a campione | completa |
| **Parafrasi del claim** | stesso contenuto, parole diverse | invariata | LLM + verifica | sì | completa |

Le famiglie della versione ridotta sono tutte automatiche e con etichetta garantita per costruzione: si fanno in pochi giorni, senza annotatori. Le famiglie "rimozione dell'evidenza" e "coppia minima" sono anche quelle che distinguono **non stabilito** da **contraddetto**, utile per RQ4.

Esempio di coppia minima sul claim:

> Documento: "Okeh Records is a subsidiary of Sony Music Entertainment and was founded by Otto Heinemann."
> Claim originale (1): "Okeh Records was founded by Otto Heinemann."
> Claim modificato (0): "Okeh Records was founded by **Thomas Edison**."

## 4. Validazione umana (versione completa)

- **Campione**: ~500 esempi delle famiglie in cui l'etichetta cambia (rimozione, coppie minime, parafrasi del claim, contesto naturale).
- **Annotatori**: 2, indipendenti, con una guida di una pagina ("il documento supporta ogni parte del claim? sì / no, contraddetto / no, non stabilito / incerto").
- **Accordo**: kappa di Cohen e accuratezza delle etichette automatiche contro il giudizio umano. Obiettivo: ≥ 95% di etichette automatiche confermate; le famiglie sotto il 90% vanno riviste o tolte.
- **Tempo**: ~500 esempi × ~40 secondi × 2 persone ≈ 11 ore in totale.

## 5. Metriche del set

| Proprietà | Metrica |
|---|---|
| Sensibilità all'evidenza | % di esempi in cui la decisione si inverte correttamente (rimozione, coppie minime) |
| Invarianza | % di decisioni invariate (distrattori, posizione, formulazione, campi, opzioni, composizione, parafrasi del claim) |
| Coerenza di polarità | AUC della forma negata contro quella positiva; % di coppie con P(sì positivo) + P(sì negato) ≈ 1 |
| Dipendenza dal solo claim | P(supportato) media con documento vuoto |
| Calibrazione sotto perturbazione | variazione dell'ECE fra originale e perturbato |

Un modello ideale ha sensibilità 100% e invarianza 100%. I due numeri insieme sono più informativi della sola accuratezza: un modello che dice sempre "no" ha invarianza alta ma sensibilità nulla.

## 6. Dimensione e formato

- Versione ridotta: ~300–500 esempi per famiglia, ~3.000 in tutto.
- Versione completa: ~300–1.000 esempi per famiglia, ~6.000–10.000 in tutto.
- Formato: JSONL con `id`, `family`, `parent_id`, `document`, `claim`, `question_variant`, `label`, `expected_change` ("flip" / "invariant" / "complement"), `source`, `license`, `validated`.
- Rilascio: dati derivati da fonti con licenza compatibile (CC BY / CC BY-SA / MIT; per le NC, solo uso non commerciale e dichiarato) + script per rigenerare il resto, incluse le trasformazioni su AggreFact.

## 7. Documentazione richiesta dai revisori

Una scheda tipo *datasheet*: motivazione, composizione, processo di raccolta, trasformazioni, validazione, accordo, usi previsti e non previsti, licenze, contatti. Va in appendice e nel repository del set.
