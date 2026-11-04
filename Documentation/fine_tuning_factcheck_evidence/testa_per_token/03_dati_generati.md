# Passo 3 — Dati generati, etichette ricavate col modello, evidenza spostata

## In breve

Le stesse tre famiglie del piano per frase ([../testa_per_frase/03_dati_generati.md](../testa_per_frase/03_dati_generati.md)), con una differenza: le etichette si salvano come **intervalli di caratteri** (`gold_char_sets`), così servono direttamente alla testa per token. Per le prime prove (SciFact, WiCE, HoVer) questo passo è facoltativo: serve per avere evidenza su testi non Wikipedia e più casi lontani.

## 1. Documenti generati con posizioni note (C2D, CG2C)

Procedura del piano per frase: il generatore scrive un articolo con le frasi date marcate `<e1>…</e1>`; si tolgono i tag e si annota dove stavano.

Per la testa per token si salva l'intervallo **esatto** del testo marcato, non la frase che lo contiene. È l'unica fonte in cui l'etichetta è precisa anche dentro una frase: se il generatore ha inserito il fatto in una frase più lunga, solo quella parte è evidenza. Queste righe sono il miglior segnale per insegnare tratti stretti (esperimento H10, [11](11_esperimenti.md)).

Varianti come nel piano per frase: il negativo (si toglie un fatto necessario) e il claim parziale (evidenza = solo i fatti usati).

## 2. Etichette ricavate col modello (ablazione)

Per righe supportate senza evidenza annotata. L'ablazione toglie **frasi** dal documento e guarda il verdetto: per farlo serve dividere il documento in unità. Si usano le frasi del nostro divisore. È un lavoro offline sui dati di training, che non entra in produzione: la dipendenza dal divisore resta fuori dal modello.

Procedura identica al piano per frase (ricerca greedy del gruppo più piccolo di frasi che basta, con KnowLine-4B, giudice LLM, accordo con elephant-1). Le frasi trovate si salvano come intervalli di caratteri in `gold_char_sets`, con `evidence_quality = ablation` e peso 0,5.

## 3. Evidenza spostata (righe sparse)

Da righe con evidenza nota e almeno 2 frasi gold: le frasi gold si spostano agli estremi del documento, o si inseriscono frasi di disturbo fra di esse, o si aggiunge testo di disturbo all'inizio finché l'evidenza è oltre 1.024 o 4.096 token. Si spostano intere le frasi della fonte (non i token), e gli intervalli di `gold_char_sets` si ricalcolano sul documento nuovo con lo stesso `DocBuilder` del builder.

Controlli del piano per frase: frasi che iniziano riferendosi alla precedente ("He", "This", "The company") non si spostano da sole; il giudice conferma l'etichetta; le frasi di disturbo vengono solo da documenti di training.

## 4. Codice

| Script | Cosa fa |
|---|---|
| `evidence/data/gen_marked.py` | C2D e CG2C con intervalli esatti |
| `evidence/data/label_ablation.py` | ablazione con KnowLine-4B, uscita in intervalli di caratteri |
| `evidence/data/augment_sparse.py` | evidenza spostata |

Tutti scrivono righe nel formato di `evidence/data/schema.py` (con `gold_char_sets`) in `dataset/evidence/`, validate come le altre.
