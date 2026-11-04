# Riepilogo — tutte le formule in una pagina

Notazione in [00](00_overview.md) §3. Fra parentesi il file e la funzione che realizzano ogni formula.

## 1. Dal documento alle feature (un forward)

**Frasi** (`sentences.split`): documento → $\{[s_k, e_k)\}_{k<n}$.

**Token delle frasi** (`render.sentence_token_spans`): con $\gamma_t$ l'ultimo carattere non spazio del token $t$, riportato al testo originale,

$$
\phi(t) = k \iff s_k \le \gamma_t < e_k, \qquad a_k = \min \phi^{-1}(k), \quad b_k = \max \phi^{-1}(k), \quad n_k = b_k - a_k + 1 .
$$

**Verdetto** (`model.p_supported`):

$$
P(\text{supportato}) = \sigma\big(\ell_A - \ell_B\big), \qquad \ell = \log\mathrm{softmax}\big(W_U\,\mathrm{RMSNorm}(H^{(31)}_T)\big).
$$

**Riassunti delle frasi** (`model.SentenceTap`), per $l \in \{19, 23, 27, 31\}$:

$$
h^{end}_{k,l} = H^{(l)}_{b_k}, \qquad h^{mean}_{k,l} = \frac{1}{n_k}\sum_{i=a_k}^{b_k} H^{(l)}_i, \qquad q_l = H^{(l)}_T .
$$

**Attenzione per frase** (`model.sdpa_capture`, `attention_per_sentence`), per $l \in \{3, 7, \ldots, 31\}$ e $j \in \{0..15\}$:

$$
A^{(l,j)}_i = \mathrm{softmax}_i\Big(\frac{\mathbf q^{(l,j)}_T\cdot \mathbf k^{(l,\lfloor j/4\rfloor)}_i}{\sqrt{256}}\Big), \qquad
att_k[(l,j)] = \sum_{i=a_k}^{b_k} A^{(l,j)}_i \quad\Rightarrow\quad att_k \in \mathbb{R}^{128}.
$$

## 2. La testa (`head.EvidenceHead`)

$$
\begin{aligned}
\text{A.}\quad & \pi = \mathrm{softmax}(\alpha), \qquad m(V) = \mathrm{LN}\big(\textstyle\sum_l \pi_l V_l\big)\\
\text{B.}\quad & e_k = W_h\big[m(h^{end}_k); m(h^{mean}_k)\big] + W_a\big[\xi_k\, att_k; \log n_k\big] + P_k + t_{frase}\\
& x_\star = W_q\, m(q) + t_{risposta}\\
\text{C.}\quad & [X_\star; X_0; \ldots; X_{n-1}] = \mathrm{Encoder}^{(2)}_{\text{non causale}}\big([x_\star; e_0; \ldots; e_{n-1}]\big)\\
\text{D.}\quad & s_k = w_o^\top X_k + b_o, \qquad s_\star = w_o^\top X_\star + b_o
\end{aligned}
$$

Il **layer dell'encoder** (pre-norm, 4 teste da 64, FFN 1.024):

$$
U = Z + \mathrm{MHA}(\mathrm{LN}_1 Z),\qquad Z' = U + W_2\,\mathrm{ReLU}(W_1\,\mathrm{LN}_2 U + b_1) + b_2 .
$$

**Selezione** (`metrics.select`), con $\theta = s_\star + \delta$:

$$
\mathcal{S} = \mathrm{top}_5\{k : s_k > \theta\}\ \text{oppure}\ \{\arg\max_k s_k\}, \quad\text{solo se } P(\text{supportato}) > 0{,}5 .
$$

**H-PAIR**: $S_{kj} = (U_1X_k)^\top(U_2X_j)$, e poi $s_k \mathrel{+}= \tau\max_{j\neq k}\big(S_{kj} + \mathrm{sg}(s_j)\big)$.

**Parametri:** 3.716.357 per la testa completa, 3.060.994 per H0. Tempo: circa 1,1 ms per documento.

## 3. Le loss (`losses.py`)

Con $d_k = s_k - s_\star$, $G$ = frasi gold e $\omega$ il peso delle non gold:

$$
\begin{aligned}
\ell_{sup} &= \log\Big(1 + \sum_{i\notin G}\omega_i e^{d_i}\Big) + \log\Big(1+\sum_{j\in G} e^{-d_j}\Big) \;\approx\; \big[\max_{i\notin G} d_i\big]_+ + \big[-\min_{j\in G} d_j\big]_+\\
\ell_{neg} &= 0{,}3\,\log\Big(1+\sum_i e^{d_i}\Big)\\
\ell_{pos} &= 0{,}3\,\Big[\log\big(1+e^{-\mathrm{LSE}_i d_i}\big) + 0{,}1\cdot\tfrac{1}{n}\textstyle\sum_i\sigma(d_i)\Big]\\
\ell_{ref} &= \frac1n\sum_k c_k\,\mathrm{BCE}(s_k, y_k) + 0{,}5\cdot\frac{1}{\lvert G\rvert}\sum_{j\in G}\mathrm{softplus}\big(1 - s_j + \max_{i\notin G}s_i\big),\quad c_k \in \{10, \omega\}\\
\mathcal{L}_{batch} &= \frac{\sum_b w_b\,\ell_b}{\sum_b w_b\,\mathbb{1}[\ell_b \text{ attivo}]}
\end{aligned}
$$

**Gradiente di ZLPR:** $\partial\ell_{sup}/\partial s_i = \omega_i e^{d_i}/(1+\sum_{i'}\omega_{i'}e^{d_{i'}})$ per $i \notin G$. È una softmax con la soglia come classe a logit 0: la non gold peggiore riceve la spinta maggiore, quelle già sotto soglia quasi niente.

## 4. Training (`train_head.py`)

$$
\eta_\kappa = 10^{-3}\cdot\min\Big(1,\frac{\kappa+1}{W}\Big)\cdot\tfrac12\Big(1+\cos\frac{\pi\min(\kappa,K)}{K}\Big),\qquad W = \lfloor 0{,}05K\rfloor
$$

$$
\text{checkpoint: } \max\ \tfrac12\big(\mathrm{F1} + \text{completo}\big)\ \text{su } \texttt{dev\_evid\_select}
$$

Regola: $\delta \in [-1, 1]$ a passi di 0,1 (ZLPR), oppure $\tau_e \in [0{,}05; 0{,}95]$ (ref).

## 5. Metriche (`metrics.py`, `eval_evidence.py`)

$$
\mathrm{P} = \frac{\lvert\mathcal S\cap G\rvert}{\lvert\mathcal S\rvert},\qquad \mathrm{R} = \frac{\lvert\mathcal S\cap G\rvert}{\lvert G\rvert},\qquad \text{completo} = \mathbb 1[\exists r: G^{(r)}\subseteq\mathcal S]
$$

$$
\text{sufficienza} = \mathbb 1[p(\mathcal S) > 0{,}5],\qquad \text{necessità} = \mathbb 1[p(\bar{\mathcal S}) \le 0{,}5]
$$

$$
\mathrm{BAcc} = \tfrac12(\mathrm{TPR}+\mathrm{TNR}),\qquad \mathrm{ECE} = \sum_c \tfrac{\lvert B_c\rvert}{N}\lvert \mathrm{acc}(B_c) - \mathrm{conf}(B_c)\rvert
$$

## 6. Dati (`make_splits.py`, `make_mix.py`, `baselines.py`)

$$
u(\textit{doc}) = \mathrm{sha256}(\texttt{"13:sale:doc\_id"})_{[0:8]} / 2^{64},\qquad
\text{negativi tenuti} = \mathrm{round}\Big(\frac{f\, P}{1-f}\Big)
$$

$$
\mathrm{BM25}_k = \sum_{w}\mathrm{IDF}(w)\,\frac{\mathrm{tf}\,(k_1+1)}{\mathrm{tf}+k_1\big(1-b+b\,\lvert S_k\rvert/\overline{\lvert S\rvert}\big)}
$$

## 7. Perché ogni scelta, in una riga

| Scelta | Perché |
|---|---|
| un solo forward, modello congelato | latenza di un decision model (+0,5% misurato); verdetto invariato |
| preambolo con il claim | nel modello causale le frasi altrimenti non sanno che cosa si verifica |
| seconda copia del documento | ha già visto tutto il documento e il claim |
| layer 19, 23, 27, 31 | subito dopo un layer pieno; metà alta della rete; non solo l'ultimo, specializzato nella risposta |
| $h^{end}$ e $h^{mean}$ | fine della frase (vede il contesto) e media (non dipende dalla punteggiatura) |
| $att_k$ | dove ha guardato il modello per decidere (E-ATT da solo: F1 0,75) |
| transformer non causale | la frase 0 vede la frase 40: evidenza sparsa |
| soglia $s_\star$ per documento, stessa $w_o$ | numero di frasi adattato al documento; stessa scala dei punteggi |
| ZLPR | hinge morbido sul caso peggiore: completo vs non completo; gradiente concentrato su chi sbaglia |
| `mil_neg` | nessuna frase accesa sui claim non supportati |
| $\omega < 1$ | annotazioni incomplete |
| split per documento con hash | nessuna fuga; stabile |
