# Venue, calendario, rigore e rischi

Le date sono quelle trovate l'8 ottobre 2026; vanno sempre ricontrollate sui siti ufficiali (ACL Rolling Review, siti delle conferenze).

## 1. Dove sottomettere

| Venue | Come | Scadenze note / attese | Adatta a | Note |
|---|---|---|---|---|
| **ACL 2027** (Kyoto, 17–22 agosto 2027) | ACL Rolling Review (ARR), poi *commitment* | l'ultimo ciclo ARR utile per ACL 2027 è a **gennaio 2027** (date esatte ancora "TBA" sul sito) | paper completo (Parte A + B) | **obiettivo consigliato** |
| EMNLP 2027 | ARR | ciclo atteso verso maggio–giugno 2027 | piano di riserva, o versione più ricca | |
| Findings (ACL / EMNLP) | stessa sottomissione ARR | decisione dei chair | se main non passa | |
| NAACL 2027, EACL 2027, COLING 2027 | ARR | i commitment (ottobre e dicembre 2026) richiedono revisioni ARR già ottenute | — | non raggiungibili per questo lavoro |
| ICLR 2027 | diretta | scadenza passata (25/09/2026) | — | |
| Workshop FEVER (Fact Extraction and VERification) | diretta | di solito a primavera, co-locato con ACL o EMNLP | sola Parte A, o versione breve | attenzione alle regole sulla doppia sottomissione se si pubblicano atti |
| TrustNLP (co-locato con ACL/NAACL) | diretta | primavera 2027, da verificare | Parte A | |
| NeurIPS 2027 Datasets & Benchmarks | diretta | attesa verso maggio 2027 | solo se si fa un dataset vero (Opzione 3) | |
| **arXiv** | preprint | subito | fissare la priorità | non impedisce la sottomissione ARR (seguire le regole ARR sull'anonimato) |

## 2. Calendario a ritroso (obiettivo: ARR gennaio 2027 per ACL 2027)

| Periodo | Lavoro | Uscita |
|---|---|---|
| ottobre 2026 (sett. 2–4) | Step 0: split, test fuori dominio; baseline di tutti i modelli con protocollo zero-shot; LLM generici con la stessa lettura (B1); checker specializzati (B3) | tabella della Parte A |
| inizio novembre 2026 | set diagnostico (costruzione + validazione di 500 esempi); analisi degli errori | Parte A completa |
| metà novembre 2026 | **preprint arXiv della Parte A** (facoltativo ma consigliato, per la priorità) | preprint |
| novembre 2026 | codice di training; replay, varianti, negativi difficili; pilota su JPT | primo risultato di fine-tuning |
| dicembre 2026 | ablazioni su JPT; Winnow; controlli sulle basi; seed | risultati della Parte B |
| fine dicembre 2026 – gennaio 2027 | scrittura, figure, appendici, revisione interna, checklist ARR (riproducibilità, limitazioni, etica) | paper completo |
| gennaio 2027 | **sottomissione ARR** | |
| primavera 2027 | revisioni ARR, risposta ai revisori, commitment ad ACL 2027 | |

Margine: è stretto ma fattibile, perché la Parte A è già in gran parte pronta. Se a metà dicembre la Parte B non è solida, si sottomette la Parte A più un pilota di fine-tuning a gennaio, e si sposta il paper completo su EMNLP 2027.

## 3. Checklist di rigore (da completare prima della sottomissione)

**Dai 14 punti dell'audit dei TDM** ([01_stato_arte_e_novita.md](01_stato_arte_e_novita.md), §4):
- [ ] B1 baseline con log-probabilità di LLM generici; B2 cascata generativa; B3 checker specializzati
- [ ] T1 dispersione fra modelli; T2 invarianza; T3 opzioni rinominate o invertite; T4 provenienza dei dati
- [ ] R1 versioni e date esatte (pesi, Jev); R2 intervalli di confidenza; R3 scelte post-valutazione dichiarate
- [ ] C1 calibrazione su dati tenuti da parte; C2 accuratezza ad alta confidenza; C3 soglie congelate prima del test; C4 latenza con e senza cache

**Dalla checklist ARR**:
- [ ] sezione Limitations (obbligatoria)
- [ ] dichiarazione su rischi ed etica
- [ ] licenze di dati e modelli, rispetto dei termini (LLM-AggreFact: solo valutazione)
- [ ] dettagli di calcolo: GPU, ore, numero di corse
- [ ] iperparametri e procedura di scelta
- [ ] codice e dati rilasciati (o motivazione)
- [ ] uso di assistenti AI dichiarato secondo le regole della venue

**Specifiche di questo lavoro**:
- [ ] dichiarare che il pacchetto di domande e la soglia di Jev nel run originale sono stati scelti sul dev di AggreFact, e il controllo intermedio sul test (`results/INTERIM_AMENDMENT.md`); nel paper usare il protocollo zero-shot
- [ ] decontaminazione documentata (RAGTruth 98%, WiCE 30%)
- [ ] nessun test usato per scegliere nulla

## 4. Rischi e contromisure

| Rischio | Probabilità | Impatto | Contromisura |
|---|---|---|---|
| Scoop da un altro gruppo (il filone produce paper ogni settimana) | media | alto | preprint della Parte A entro metà novembre; ricontrollare arXiv ogni settimana |
| Revisori: "è solo benchmarking" | alta per la sola Parte A | alto | Parte B + RQ2 (contro le basi generiche) + analisi di calibrazione e robustezza |
| Il fine-tuning non batte i checker specializzati | media | medio | il paper regge come analisi se B-ctrl e oblio sono misurati; raccontare il compromesso |
| Jev cambia versione o prezzo | media | basso | conclusioni principali sui modelli aperti; versione e date registrate; cache delle risposte rilasciate |
| Etichette rumorose (ExpertQA ~59 per tutti; HaluEval già segnalato) | certa | medio | analisi degli errori con categoria "etichetta dubbia"; non inseguire le fonti rumorose |
| Licenze (JPT non commerciale; dati NC) | certa | basso per un paper | dichiarare; rilasciare adapter con le licenze corrette |
| Calcolo insufficiente | media | medio | JPT con QLoRA sul portatile; Winnow solo per la configurazione finale; L40S per le corse critiche |
| Doppia sottomissione (workshop con atti + conferenza) | bassa | alto | preferire arXiv; se workshop, scegliere uno senza atti o un contenuto chiaramente diverso |

## 5. Etica e limitazioni (bozza)

- Un fact checker documento–claim verifica il **supporto nel documento**, non la verità nel mondo: un documento sbagliato produce verdetti sbagliati. Va scritto chiaramente, per evitare usi impropri come "rilevatore di fake news".
- Tutti i dati sono in inglese; nessuna garanzia su altre lingue.
- Le probabilità calibrate su un dominio possono non esserlo su un altro (lo misuriamo: RQ3).
- Uso di modelli chiusi (Jev) e riproducibilità limitata dalle loro versioni.
- I dati generati con LLM possono contenere errori sistematici del generatore; sono verificati da un secondo modello e su un campione umano.
