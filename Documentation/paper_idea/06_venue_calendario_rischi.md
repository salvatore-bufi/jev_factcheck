# Venue, calendario, rigore e rischi

Date verificate il 26 ottobre 2026 sui siti ufficiali dove indicato; le altre vanno ricontrollate.

## 1. Dove sottomettere

| Venue | Come | Scadenze | Adatta a | Note |
|---|---|---|---|---|
| **ACL 2027** (Kyoto, 17–22 agosto 2027) | ACL Rolling Review, poi *commitment* | **ARR: 4 gennaio 2027**, 23:59 AoE (ufficiale, [call](https://2027.aclweb.org/calls/main/)); commitment e notifiche ancora da annunciare | **primo paper** (analisi + B-ctrl) | **obiettivo consigliato**. Long paper 8 pagine (9 nella versione finale); Limitations e Responsible NLP checklist obbligatorie; nessuna doppia sottomissione |
| Findings of ACL 2027 | stessa sottomissione | decisione dei chair | se main non passa | |
| EMNLP 2027 | ARR | conferenza a inizio novembre 2027 (Messico o America Centrale, provvisorio); ciclo ARR non ancora annunciato | **secondo paper** (metodo completo + set diagnostico completo) | |
| NAACL 2027, COLING 2027, EACL 2027 | ARR | scadenze ARR del 12 ottobre 2026 passate | — | non raggiungibili |
| ICLR 2027 | diretta | passata | — | |
| Workshop FEVER | diretta | FEVER 2026 si è tenuto il 28 marzo 2026 a Rabat; l'edizione 2027 non è ancora annunciata | versione breve, o il set diagnostico | attenzione alla doppia sottomissione se ci sono atti |
| NeurIPS 2027 Datasets & Benchmarks | diretta | attesa verso maggio 2027 | solo per un dataset vero (etichette distribuzionali) | |
| **arXiv** | preprint | **entro fine novembre 2026** | fissare la priorità | ARR permette i preprint (da verificare la regola corrente sull'anonimato nella call); la sottomissione resta anonima |

Il tema speciale di ACL 2027 è "Homogenization and knowledge collapse in LLMs". C'è un aggancio possibile ma da non forzare: se i TDM costruiti sulla stessa base sbagliano negli stessi punti (A9), è un caso misurabile di omogeneizzazione. Va citato solo se i dati lo mostrano.

## 2. Calendario a ritroso (obiettivo: ARR 4 gennaio 2027)

Dieci settimane. Il dettaglio esperimento per esperimento è in [11](11_design_esperimenti.md) §9. La parte di analisi viene prima, perché serve per il preprint e perché è il paper anche se B-ctrl non arriva.

| Periodo | Lavoro | Uscita |
|---|---|---|
| 26 ott – 8 nov | Step 0: split, dev, test fuori dominio. Run completi dei modelli con adapter pronto (d3-flash, StartLux-9B, Nimble-9B-v3, MiniCheck-7B). Adapter delle basi (lettura ingenua e calibrata). Jev 1.13 e 1.14 (riferimento) con il protocollo zero-shot | tabella A1 parziale |
| 9 – 22 nov | Adapter nuovi (Clef-flash, Tacit, pplx-decider-flash, Kev-9B/4B, Strands 2B). Inferenza fuori dominio. Varianti di protocollo (RQ3) e set diagnostico ridotto. Codice di B-ctrl e pilota su d1-3B | A1–A3 complete |
| **30 nov** | **congelamento della lista dei modelli**; **preprint arXiv** con A1–A3 (RQ1–RQ3) | preprint |
| 1 – 14 dic | B-ctrl sulla coppia 9B; calibrazione e trasferimento (A5); errori e accordo (A4, A8, A9) | risultati completi |
| **12 dic** | **punto di decisione**: se B-ctrl non è solido, resta il pilota in una sottosezione o in appendice | |
| 15 – 31 dic | scrittura, figure, appendici, revisione interna, checklist ARR | paper completo |
| 1 – 4 gen | ultimi controlli, sottomissione | |
| primavera 2027 | revisioni ARR, risposta, commitment ad ACL 2027 | |

Margine: stretto, ma il vincolo dei 9B lo rende più sicuro: ogni modello entra in una GPU da 24–48 GB, quindi più run possono andare in parallelo su GPU diverse. La parte più a rischio sono gli adapter nuovi (runtime proprietari); se mancano, RQ2 si appoggia ai cinque TDM su Qwen3.5-9B già pronti o fatti.

## 3. Checklist di rigore (da completare prima della sottomissione)

**Dai 14 punti dell'audit dei TDM** ([01_stato_arte_e_novita.md](01_stato_arte_e_novita.md), §4):
- [ ] B1 basi con lettura ingenua **e** calibrata senza training; B2 cascata generativa con controllo a escalation casuale; B3 checker specializzati
- [ ] T1 dispersione fra modelli; T2 invarianza (polarità, composizione, distrattori, posizione); T3 opzioni rinominate o invertite; T4 provenienza e contaminazione
- [ ] R1 versioni e date esatte (revisioni dei pesi in `run.json`, Jev 1.13.0 e 1.14.0); R2 intervalli di confidenza; R3 scelte post-valutazione dichiarate
- [ ] C1 calibrazione su dati tenuti da parte; C2 accuratezza ad alta confidenza; C3 soglie congelate prima del test; C4 latenza con e senza cache

**Dalla checklist ARR**:
- [ ] sezione Limitations
- [ ] dichiarazione su rischi ed etica
- [ ] licenze di dati e modelli e rispetto dei termini (LLM-AggreFact: CC BY-ND, solo valutazione; JPT, Nimble, StartLux, Bespoke-MiniCheck: non commerciali)
- [ ] dettagli di calcolo: GPU, ore, numero di corse, energia (già misurata dal benchmark)
- [ ] iperparametri e procedura di scelta
- [ ] codice e dati rilasciati (o motivazione)
- [ ] uso di assistenti AI dichiarato secondo la policy "Writing Assistance"

**Specifiche di questo lavoro**:
- [ ] dichiarare che il pacchetto di domande e la soglia del run originale di Jev sono stati scelti sul dev di AggreFact, e il controllo intermedio sul test ([../../legacy/results/INTERIM_AMENDMENT.md](../../legacy/results/INTERIM_AMENDMENT.md)); nel paper usare il protocollo zero-shot
- [ ] riprodurre il numero di Jev di [2609.37647](https://arxiv.org/abs/2609.37647) (78,6 a soglia 0,5) come controllo del nostro protocollo
- [ ] la grid search sul test è esplorativa: nessuna scelta del paper viene da lì
- [ ] media di AggreFact riportata anche senza RAGTruth (Decision Index); HoVer mai usato come test fuori dominio per i TDM
- [ ] decontaminazione dei dati di B-ctrl documentata (RAGTruth 98%, WiCE 30%)
- [ ] data di congelamento della lista dei modelli dichiarata

## 4. Rischi e contromisure

| Rischio | Probabilità | Impatto | Contromisura |
|---|---|---|---|
| Scoop: un lavoro con molti TDM aperti su AggreFact (tre lavori vicini in sei settimane; i produttori inviano alla leaderboard ogni settimana) | **alta** | alto | preprint entro il 30 novembre; ricerca settimanale su arXiv ([01](01_stato_arte_e_novita.md) §6); puntare su RQ2 e RQ3, che sono più difficili da replicare senza la nostra infrastruttura |
| Revisori: "è solo benchmarking" | media | alto | RQ2 e RQ3 come contributi principali; ogni tabella risponde a una domanda; B-ctrl |
| RQ2 dà differenze piccole fra base calibrata e TDM | media | basso | è un risultato: conferma l'audit sul grounding e sposta il valore dei TDM su costo e interfaccia |
| Bersaglio mobile (nuovi TDM, Jev 1.14, leaderboard che cambia) | certa | medio | data di congelamento; i nuovi modelli in appendice; il messaggio di RQ3 non dipende dai modelli |
| Contaminazione dei TDM (dati di training non pubblici; RAGTruth nella Decision Index) | alta | medio | analisi con e senza RAGTruth; test fuori dominio scelti fuori dalla Decision Index; dichiarato nelle limitazioni |
| Adapter nuovi lenti da scrivere (runtime proprietari, versioni di vLLM incompatibili) | media | medio | priorità ai modelli già pronti; Clef ha un `systemone` in Python nel repository dei pesi |
| Revisori: "è un confronto incentrato su Qwen" | alta | medio | quota di Qwen nell'ecosistema documentata; blocco non Qwen (Gemma-4-E4B, Ministral-3-8B, LFM2.5, encoder) con E2b; B-ctrl replicato su Gemma-4-E4B |
| Revisori: "perché solo fino a 9B? I migliori sono 27B" | media | medio | motivazione esplicita (costo, deployment su una GPU, fine-tuning accessibile, gruppi per base più ricchi); i 27B e Jev riportati come riferimento; la distanza dai 27B è essa stessa un risultato |
| Etichette rumorose (ExpertQA ~59–61 per tutti; HaluEval segnalato) | certa | medio | categoria "etichetta dubbia" nell'analisi degli errori; non inseguire le fonti rumorose |
| Licenze (molti TDM non commerciali; AggreFact CC BY-ND) | certa | basso per un paper | dichiarare; non ridistribuire derivati di AggreFact |
| Doppia sottomissione (workshop con atti + conferenza) | bassa | alto | preferire arXiv |

## 5. Etica e limitazioni (bozza)

- Un fact checker documento–claim verifica il **supporto nel documento**, non la verità nel mondo: un documento sbagliato produce verdetti sbagliati. Va scritto chiaramente, per evitare usi come "rilevatore di fake news".
- La BAcc nasconde l'asimmetria degli errori: un guardiano che ammette più claim non supportati è più pericoloso a parità di BAcc. Lo misuriamo e lo diciamo.
- Tutti i dati sono in inglese.
- Le probabilità calibrate in un dominio possono non esserlo in un altro (RQ5).
- Uso di modelli chiusi e di TDM aperti con dati di training non pubblici: contaminazione non esclusa.
- I dati generati con LLM (set diagnostico, B-ctrl) possono contenere errori sistematici del generatore; sono verificati automaticamente e, nella versione completa, su un campione umano.
