# Il set diagnostico (contributo secondario)

## 1. Perché un set diagnostico e non un dataset nuovo

Un benchmark come LLM-AggreFact dice **quanto** un modello sbaglia, non **perché**. I typed decision model promettono proprietà precise: leggono il documento, rispondono in modo stabile alla stessa domanda posta in modi diversi, danno probabilità calibrate. Un set diagnostico misura esattamente queste proprietà, con esempi costruiti in modo controllato. Costa una frazione di un dataset nuovo, perché parte da esempi già etichettati.

## 2. Da dove si parte

Solo da **test** di dataset con licenza compatibile e mai usati in training:
- test in distribuzione delle nostre fonti (FEVER `paper_test`, HoVer, VitaminC test, MiniCheck e SummEdits nella parte di test);
- fonti fuori dominio (SciFact, Climate-FEVER, HealthVer).

**Non** da LLM-AggreFact: la sua licenza permette la valutazione, ma derivare e ridistribuire un nuovo dataset dalle sue parti va verificato fonte per fonte. Se si vuole includerlo, distribuire solo gli script di trasformazione, non i testi.

## 3. Le famiglie di test

| Famiglia | Trasformazione | Etichetta attesa | Costruzione | Validazione umana |
|---|---|---|---|---|
| **Rimozione dell'evidenza** | si cancellano dal documento le frasi gold che supportano il claim | da 1 a 0 (non stabilito) | automatica dove l'evidenza è annotata (FEVER, HoVer); con LLM altrove | sì: il documento non deve più supportare il claim, né contraddirlo |
| **Coppia minima sul claim** | si cambia un numero, una data, un'entità o un'attribuzione | da 1 a 0 (contraddetto) | LLM (`contradict_claim`) + verifica automatica | sì |
| **Coppia minima sul documento** | si cambia nel documento il valore che il claim cita | da 1 a 0 | VitaminC (già contrastivo) + LLM | a campione |
| **Distrattori** | paragrafi irrilevanti (da altri documenti dello stesso dominio) prima e dopo | invariata | automatica | no (l'etichetta non cambia per costruzione) |
| **Posizione dell'evidenza** | la frase di supporto messa all'inizio, a metà o alla fine di un documento lungo | invariata | automatica | no |
| **Solo claim** | documento vuoto o non pertinente | 0 | automatica | no |
| **Formulazione della domanda** | 10 parafrasi mai viste in training | invariata | LLM + filtro | a campione |
| **Campi rinominati** | `document`/`claim` → `passage`/`assertion`, `context`/`answer`, ... | invariata | automatica | no |
| **Paraphrase del claim** | stesso contenuto, parole diverse | invariata | LLM + verifica | sì |

Esempio di coppia minima sul claim:

> Documento: "Okeh Records is a subsidiary of Sony Music Entertainment and was founded by Otto Heinemann."
> Claim originale (1): "Okeh Records was founded by Otto Heinemann."
> Claim modificato (0): "Okeh Records was founded by **Thomas Edison**."

## 4. Validazione umana

- **Campione**: ~500 esempi delle famiglie in cui l'etichetta cambia (rimozione, coppie minime, parafrasi del claim).
- **Annotatori**: 2, indipendenti, con una guida di una pagina ("il documento supporta ogni parte del claim? sì / no / incerto").
- **Accordo**: kappa di Cohen fra annotatori e accuratezza delle etichette automatiche contro il giudizio umano. Obiettivo: ≥ 95% di etichette automatiche confermate; le famiglie sotto il 90% vanno riviste o tolte.
- **Tempo**: ~500 esempi × ~40 secondi × 2 persone ≈ 11 ore di annotazione in totale.

## 5. Metriche del set

| Proprietà | Metrica |
|---|---|
| Sensibilità all'evidenza | % di esempi in cui la decisione si inverte correttamente (rimozione, coppie minime) |
| Invarianza | % di decisioni invariate (distrattori, posizione, formulazione, campi, parafrasi del claim) |
| Dipendenza dal solo claim | P(supportato) media con documento vuoto |
| Calibrazione sotto perturbazione | variazione dell'ECE fra originale e perturbato |

Un modello ideale ha sensibilità 100% e invarianza 100%. I due numeri insieme sono più informativi della sola accuratezza: un modello che dice sempre "no" ha invarianza alta ma sensibilità nulla.

## 6. Dimensione e formato

- ~300–1.000 esempi per famiglia, ~5.000–8.000 in tutto.
- Formato: JSONL con `id`, `family`, `parent_id`, `document`, `claim`, `label`, `expected_change` ("flip" / "invariant"), `source`, `license`, `validated` (true/false).
- Rilascio: dati derivati da fonti con licenza compatibile (CC BY / CC BY-SA / MIT; per le NC, solo uso non commerciale e dichiarato) + script per rigenerare il resto.

## 7. Documentazione richiesta dai revisori

Una scheda tipo *datasheet*: motivazione, composizione, processo di raccolta, trasformazioni, validazione, accordo, usi previsti e non previsti, licenze, contatti. Va in appendice del paper e nel repository del set.
