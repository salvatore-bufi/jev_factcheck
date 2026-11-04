# Passo 5 — Il prompt con il preambolo

Identico al piano per frase: [../testa_per_frase/05_formato_prompt.md](../testa_per_frase/05_formato_prompt.md). Riassunto:

- si aggiunge `Claim under review: <claim>` e una riga vuota **prima dello stato**, nello stesso turno; il resto del prompt è quello del piano binario, prodotto dal codice degli autori;
- si sceglie il testo del preambolo (P-A o P-B) misurando il verdetto su `dev_select` con e senza preambolo;
- se il verdetto del piano binario resta entro 0,3 punti di BAcc e AUC con il preambolo, lo si riusa; altrimenti lo si riaddestra con il preambolo;
- la scelta si scrive in `question.json` (`preamble`, `verdict`).

Due note specifiche della testa per token:

1. **KnowLine scrive due volte le richieste fino a 3.000 token.** I token del documento si prendono dalla **seconda** copia, che ha già visto il claim e l'intero documento della prima. Il preambolo fa parte dello stato e compare in entrambe le copie.
2. **Nei modelli bidirezionali** (elephant-1 e altri) i token del documento vedono il claim negli 8 layer pieni anche senza preambolo; l'esperimento H8 ([11](11_esperimenti.md)) misura quanto serve.
