# Una strada da esplorare: etichette per token dal modello stesso

## Il problema

Le etichette della testa per token sono grossolane: le fonti segnano frasi intere, quindi dentro una frase gold anche le parole irrilevanti hanno etichetta 1. La loss compensa (chiede di colpire ogni frase gold, non di accenderla tutta), ma la testa non riceve mai un segnale su **quale parte** della frase conta. Il vantaggio principale della testa per token, evidenziare solo ciò che serve, resta quindi poco sfruttato.

## L'idea

Ricavare etichette più fini **dal decision model stesso**, offline, togliendo pezzi più piccoli di una frase e misurando quanto cambia il verdetto:

```text
per ogni frase gold g di una riga positiva:
    si divide g in pezzi fissi di ~5 token (nessun divisore: solo posizioni)
    per ogni pezzo w:  importanza(w) = z(documento) − z(documento senza w)        z = margine del verdetto (sì − no)
```

I pezzi con importanza alta sono la parte della frase che regge il verdetto (il nome, la data, la relazione); quelli con importanza vicina a zero sono contorno. Le etichette della testa diventano morbide: 1 sui pezzi importanti, un valore più basso (0,3) sugli altri token della frase gold, 0 fuori.

## Come farlo

1. **Righe**: i positivi di training con frasi gold (HoVer `exact`, SciFact, WiCE train).
2. **Calcolo** (`evidence/label_token_importance.py`), con KnowLine-4B: un forward per pezzo, solo dentro le frasi gold (in media 3 frasi × ~6 pezzi = ~18 forward per riga). KnowLine è causale: i prompt sono uguali fino al pezzo tolto, quindi si riusa il calcolo della parte iniziale comune.
3. **Prima verifica, senza training**: su 50 righe lette a mano, i pezzi con importanza più alta sono davvero la parte che dimostra il claim? Se no (il modello segue scorciatoie, o l'importanza è piatta), ci si ferma.
4. **Training** (esperimento H-IMP, fase 1a): `L_tok` usa le etichette morbide al posto di quelle per frase; `L_hit` e `L_out` restano uguali.
5. **Valutazione**: le metriche del passo 10, più la **lunghezza media dei tratti a parità di frasi gold colpite** (più corta = più precisa) e la sufficienza (i tratti più corti reggono ancora il verdetto?).

## Quando tenerlo

Rispetto alla configurazione migliore senza H-IMP:
- frasi gold colpite e insieme completo non più di 1 punto sotto;
- tratti almeno del 30% più corti;
- sufficienza non più di 2 punti sotto.

## Rischi

| Rischio | Cosa fare |
|---|---|
| Togliere 5 token rende la frase sgrammaticata e il modello reagisce alla forma, non al contenuto | confrontare con la sostituzione dello stesso pezzo con "…"; tenere l'importanza solo dove le due misure concordano |
| Informazione ripetuta nella frase: importanza bassa per tutti i pezzi | le etichette morbide non scendono mai sotto 0,3 dentro la frase gold |
| Costo: ~18 forward per riga | solo frasi gold, riuso della parte comune del prompt, KnowLine-4B |

## Perché vale la pena

È il modo più diretto di usare ciò che la testa per token sa fare e una testa per frase no: mostrare la parte della frase che conta. Non servono nuove annotazioni, il lavoro è tutto offline e in produzione non cambia nulla.
