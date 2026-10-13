# Validation error review: all 88 sampled cases

FP = gold unsupported, accepted. FN = gold supported, rejected. Scores are support / no unsupported detail / no contradiction.

Interpretations are manual hypotheses. Gold labels are unchanged. Case 24 received targeted rather than full-document review.

## 1. dev:227 — AggreFact-CNN — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Serge Gnabry has not featured for Arsenal's first team since March 2014 . The 19-year-old suffered a serious knee injury against Bayern Munich . He played 90 minutes for Arsenal Under 21s against Reading on Monday . The midfielder says he is feeling 'better and better' all the time .

Scores: 0.64 / 0.16 / 0.53; combined **0.16**.

The reference links a long absence to a knee injury but does not say the injury occurred against Bayern. The gold-supported summary adds that timing.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:1)

## 2. dev:304 — AggreFact-CNN — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: mike tindall 's horse monbeg dude finished third in the grand national . many clouds and aspell won the race at aintree on sunday . tindal and his wife zara philipps were at the national to watch the race . the former england rugby captain owns the horse .

Scores: 0.87 / 0.12 / 0.78; combined **0.12**.

The horse's third place and owners are supported, but the claim adds Sunday, which the supplied article does not establish.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:2)

## 3. dev:122 — AggreFact-CNN — FN

Cluster: **Strong apparent label/evidence conflicts**. Assessment: `label_review`.

Claim: Gianluigi Buffon won his 147th cap against England on Tuesday night . England ace Joe Hart has labelled Buffon a ` legend of the game ' The 37-year-old also claimed his 50th cap for his country on Tuesday . Hart is 10 years younger than Buffon .

Scores: 0.18 / 0.53 / 0.18; combined **0.18**.

Gold says supported, but the claim gives the 37-year-old the 50th cap. The document gives Buffon age 37 and 147 caps, and Hart 50 caps and an age ten years younger.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:3)

## 4. dev:156 — AggreFact-CNN — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: harry kane is nominated for both the pfa player and young player of the season awards . the tottenham striker has scored 20 premier league goals this season . kane also made his england debut, scoring against switzerland .

Scores: 0.38 / 0.04 / 0.75; combined **0.04**.

The source supports Kane's goals and debut but never names Switzerland as his debut opponent; that extra named detail makes the gold-positive label questionable under document-only support.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:4)

## 5. dev:319 — AggreFact-CNN — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: An NPR report on the unrest in Baltimore focused on tensions between blacks and Asians. Ruben Navarrette: There's little evidence that Asian businesses were targeted out of racial animus. He says the trope of widespread hostility between Asian and black communities first took root in the 1980s. Navanrette: The mainstream media continues to pit minority groups against one another.

Scores: 0.95 / 0.66 / 0.91; combined **0.66**.

The summary attaches the author's views to Ruben Navarrette, a name absent from the supplied reference. High agreement on the rest of the article masks the unestablished attribution.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:5)

## 6. dev:334 — AggreFact-CNN — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Bayern Munich travel to Porto for Champions League quarter - final first leg. Bastian Schweinsteiger and Franck Ribery are both out with injuries. Pep Guardiola's side are 10 points clear at the top of the Bundesliga. Bayern are also through to the German Cup semi - finals.

Scores: 0.85 / 0.59 / 0.61; combined **0.59**.

The summary says both absences are due to injuries. The main text assigns Schweinsteiger a virus, although an injury photo caption supplies a distracting competing cue.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:6)

## 7. dev:394 — AggreFact-CNN — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: In Baltimore unrest, NPR reported tension between African - Americans and Asian - owned businesses. Kevin Coval: It's time to call this persistent meme what it is: A misleading, hyperbolic and dangerous distraction.

Scores: 0.89 / 0.45 / 0.86; combined **0.45**.

The quotation is in the source, but the claimed speaker Kevin Coval is not identified there. Matching quotation content does not establish speaker identity.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:7)

## 8. dev:296 — AggreFact-CNN — FP

Cluster: **Strong apparent label/evidence conflicts**. Assessment: `label_review`.

Claim: watford can be promoted if they win and one of bournemouth or middlesbrough lose and norwich fail to win . rotherham 's desire to stave off relegation to league one has been made more difficult by a points deduction . derby can secure a play-off spot if they beat millwall .

Scores: 0.98 / 0.85 / 0.92; combined **0.85**.

All three promotion/relegation statements are directly repeated in the reference, yet the dataset label is unsupported. This is a strong candidate for label or provenance review.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:8)

## 9. dev:1127 — AggreFact-XSum — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: A man has died after the car he was driving left the road and crashed into a tree in Surrey.

Scores: 0.16 / 0.03 / 0.67; combined **0.03**.

The short article describes a car leaving the road but does not mention a death or a tree, both asserted by the gold-positive claim. Missing article context is a plausible explanation.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:9)

## 10. dev:874 — AggreFact-XSum — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: The US presidential election campaign is in full swing, with Hillary Clinton and Bernie Sanders locked in a tight race for the Democratic nomination.

Scores: 0.79 / 0.28 / 0.65; combined **0.28**.

The source describes an active competitive campaign but also a 263-delegate lead; whether tight race is a faithful summary is interpretive. Only the unsupported-detail check vetoes it.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:10)

## 11. dev:1157 — AggreFact-XSum — FN

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: A man who was arrested after taking what was thought to be a new form of MDMA, known as "pink champagne", may have taken a synthetic version of the drug Spice, Greater Manchester Police have said.

Scores: 0.83 / 0.24 / 0.58; combined **0.24**.

The claim combines a man who took the drug with a man arrested for supplying it. Both are 26, but the text leaves the exact linkage and timing implicit.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:11)

## 12. dev:951 — AggreFact-XSum — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Former Scotland boss Alex McLeish has agreed a deal to become the new coach of Egyptian club Zamalek, the club have confirmed.

Scores: 0.80 / 0.21 / 0.67; combined **0.21**.

McLeish signing in principle is supported; the claim additionally calls him a former Scotland boss, while the document calls him a former Scotland defender and lists other management jobs.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:12)

## 13. dev:471 — AggreFact-XSum — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `ambiguous`.

Claim: Oil prices have risen above $ 50 a barrel for the first time this year.

Scores: 0.82 / 0.32 / 0.73; combined **0.32**.

The claim's first crossing of $50 this year is a reasonable inference from a May report saying the highest level since November. The negative label may apply a stricter temporal-evidence standard.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:13)

## 14. dev:477 — AggreFact-XSum — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: Hsbc has appointed the chief executive of asian insurer aia, alex tucker, as its new chairman.

Scores: 0.93 / 0.42 / 0.83; combined **0.42**.

The source only identifies Mr Tucker, whereas the claim supplies Alex Tucker. The checker accepts the matching appointment and surname without establishing the first name.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:14)

## 15. dev:537 — AggreFact-XSum — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `likely_model_error`.

Claim: England won the women's six nations for the first time with a comfortable victory over ireland in dublin.

Scores: 0.40 / 0.31 / 0.30; combined **0.30**.

The document says first Six Nations title since 2012, whereas the claim says first time ever. A floating-point value of 0.30000000000000004 also causes acceptance at the nominal boundary.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:15)

## 16. dev:532 — AggreFact-XSum — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: The bbc should have a dedicated books programme, according to author neil harris.

Scores: 0.94 / 0.74 / 0.90; combined **0.74**.

The source says Harris and describes his books, while the claim supplies Neil Harris. The extra first name is not established by the source, despite strong agreement on the quoted view.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:16)

## 17. dev:7105 — ClaimVerify — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Shiitake mushrooms are also safe to eat raw and have a slight vanilla taste.

Scores: 0.71 / 0.20 / 0.59; combined **0.20**.

The source describes tasting raw shiitakes but also recommends cooking mushrooms and discusses risks. Whether that establishes safe to eat raw is debatable; rejection reflects the stronger safety claim.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:17)

## 18. dev:6704 — ClaimVerify — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: This heat causes some water molecules to move fast enough to escape into the air.

Scores: 0.57 / 0.19 / 0.41; combined **0.19**.

Evaporation and fast-moving molecules are supported, but this heat has no established antecedent and the snippet says no additional energy source is required.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:18)

## 19. dev:7334 — ClaimVerify — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `likely_model_error`.

Claim: The driver's seat is placed on either the left or right side of a car depending on which side of the road the vehicle is driven on.

Scores: 0.74 / 0.28 / 0.73; combined **0.28**.

The relationship between driver's seating position and road side follows from the wagon and Model T examples. The unsupported-detail question rejects the generalized paraphrase.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:19)

## 20. dev:6655 — ClaimVerify — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Keep the litterbox very clean, and do not punish the cat or confine her to just one room.

Scores: 0.58 / 0.11 / 0.85; combined **0.11**.

Cleaning the litterbox and avoiding punishment are supported. The additional instruction not to confine the cat to one room is absent from the full supplied article.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:20)

## 21. dev:7685 — ClaimVerify — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: However, some critics see smoking bans as a violation on one’s personal liberty and argue that it should be a personal choice.

Scores: 0.97 / 0.67 / 0.95; combined **0.67**.

The document expressly discusses personal liberty and the freedom to smoke. The accepted paraphrase appears supported, making the negative label worth review rather than demonstrating a clear model mistake.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:21)

## 22. dev:7548 — ClaimVerify — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: Some argue that it would improve accessibility, especially for remote voters, while others believe that it is not yet secure enough for high-stakes elections.

Scores: 0.85 / 0.34 / 0.86; combined **0.34**.

Voting from home for convenience and lack of readiness for high-stakes elections are both stated. The negative label may hinge on the added emphasis on remote voters.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:22)

## 23. dev:7038 — ClaimVerify — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `likely_model_error`.

Claim: Others argue that athletes should be paid more because they are entertaining.

Scores: 0.88 / 0.63 / 0.89; combined **0.63**.

The source describes equal pay for female athletes relative to male athletes. The claim drops that population and comparison, turning it into higher pay for athletes generally.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:23)

## 24. dev:7564 — ClaimVerify — FP

Cluster: **Broad interpretive claim; unresolved**. Assessment: `unresolved`.

Claim: Yes, the history of artistic style can be more than a story of changing preferences.

Scores: 0.86 / 0.72 / 0.89; combined **0.72**.

The claim is a broad interpretation of artistic history. The abstract and style-related paragraphs discuss cultural, environmental and technical influences, but this is not a crisp factual proposition. Only targeted sections of the 68,459-character source were inspected.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:24)

## 25. dev:12148 — ExpertQA — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: In the 1980s, it was difficult for people in townships to obtain title deeds for various reasons, primarily due to apartheid-era policies and legislation in South Africa .

Scores: 0.15 / 0.02 / 0.92; combined **0.02**.

The supplied apartheid snippet says nothing about township title deeds or barriers to obtaining them. A supported label is difficult to justify from the provided evidence alone.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:25)

## 26. dev:9462 — ExpertQA — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: These benefits include the reduction of potential cross-contamination, efficient use of resources, consistency in results, and ease of quality control analysis  .

Scores: 0.07 / 0.03 / 0.91; combined **0.03**.

The business QA/QC snippet does not establish laboratory cross-contamination, resource efficiency, or consistency of results. The gold-supported claim appears to require missing evidence.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:26)

## 27. dev:12420 — ExpertQA — FN

Cluster: **Numbers, units and precision**. Assessment: `evidence_gap`.

Claim: Africa is a continent with **54 countries** and **41 currencies**.

Scores: 0.18 / 0.05 / 0.50; combined **0.05**.

The document has 54 country entries, but its listed currency codes do not directly establish the asserted 41 currencies. Counting and defining unique official currencies is required.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:27)

## 28. dev:9404 — ExpertQA — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `likely_model_error`.

Claim: The criteria for selecting the Pritzker Architecture Prize winner mainly focus on the architect's significant contributions to humanity and the built environment through the art of architecture .

Scores: 0.81 / 0.27 / 0.87; combined **0.27**.

Honoring lasting contributions to humanity through architecture is directly supported. The added built-environment phrasing appears to trigger the unsupported-detail veto despite a close paraphrase.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:28)

## 29. dev:12363 — ExpertQA — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: Thus, it seems necessary to provide a care package, such as a Home Care Package Level 4, that supports people with high care needs and offers support in personal care like assistance with showering and dressing due to poor mobility and high falls risk .

Scores: 0.95 / 0.73 / 0.94; combined **0.73**.

A specific person's care example supports service capabilities, but the claim recommends a package for an unspecified case. The missing patient/context limits whether that recommendation follows.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:29)

## 30. dev:9288 — ExpertQA — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: This modality also has the advantage of providing both functional and anatomic information in a single examination .

Scores: 0.95 / 0.70 / 0.94; combined **0.70**.

The reference supports anato-functional information for one modality, but this modality in the claim has no explicit antecedent. The original question could determine whether the match is correct.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:30)

## 31. dev:12485 — ExpertQA — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `label_review`.

Claim: Some sources claim that **nurture** has the upper hand and that serial killers experience trauma or environmental influences that affect their development.

Scores: 0.96 / 0.81 / 0.93; combined **0.81**.

The combined source alternates between uncertainty and explicit claims that nurture is more important. The claim is qualified as some sources claim, so its acceptance is defensible despite the negative label.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:31)

## 32. dev:10839 — ExpertQA — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: They check for clarity, consistency, and correctness of the translated text, making sure it conveys the original meaning and intent while maintaining stylistic and tonal appropriateness .

Scores: 0.88 / 0.31 / 0.91; combined **0.31**.

Proofreading for clarity, style and meaning is supported, but they lacks an antecedent and tonal appropriateness is less explicit. The negative label cannot be confidently attributed to one defect.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:32)

## 33. dev:8385 — FactCheck-GPT — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Gathering additional information about small datasets for forecasting can be done through public data sources.

Scores: 0.06 / 0.03 / 0.86; combined **0.03**.

The reference discusses collecting and reducing data, but never mentions public data sources. The gold-positive claim adds a source of data absent from the snippet.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:33)

## 34. dev:8012 — FactCheck-GPT — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Single-cell multiomics provides a higher resolution understanding of cellular functions and interactions.

Scores: 0.70 / 0.15 / 0.91; combined **0.15**.

Cell-type-specific gene regulation is broadened to cellular functions and interactions. This may be an acceptable scientific summary or an unsupported expansion; the detail question is the only blocker.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:34)

## 35. dev:9219 — FactCheck-GPT — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Sheep keepers may paint their sheep to indicate their sex.

Scores: 0.12 / 0.14 / 0.85; combined **0.12**.

The reference explains paint identifying mating history, timing and rams, rather than marking sheep's sex. The positive label appears to need evidence not supplied here.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:35)

## 36. dev:8447 — FactCheck-GPT — FN

Cluster: **Strong apparent label/evidence conflicts**. Assessment: `label_review`.

Claim: The SM U-30 German submarine was commissioned on December 22, 1914.

Scores: 0.01 / 0.04 / 0.02; combined **0.01**.

Gold says supported for commissioning on December 22, 1914, while the reference explicitly gives August 26, 1914. The checker's rejection agrees with the supplied reference.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:36)

## 37. dev:8071 — FactCheck-GPT — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `ambiguous`.

Claim: Airplane glide ratio is expressed as a ratio.

Scores: 0.73 / 0.37 / 0.93; combined **0.37**.

The claim that a glide ratio is a ratio is nearly definitional, though the excerpt never gives its mathematical form. This is an entailment-standard boundary rather than a clear factual hallucination.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:37)

## 38. dev:8821 — FactCheck-GPT — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: ECharts Java is a library.

Scores: 0.94 / 0.84 / 0.95; combined **0.84**.

The Chinese snippet describes a Java visualization library and names ECharts-Java. The accepted English claim appears reasonable; the negative label deserves multilingual evidence review.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:38)

## 39. dev:9053 — FactCheck-GPT — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: In the plot of The Night Buffalo, Manuel starts to discover some disturbing and perplexing occurrences connected to Gregorio.

Scores: 0.94 / 0.56 / 0.92; combined **0.56**.

Mysterious mailings tied to Gregorio and increasingly strange situations support the claim's plot summary. The excerpt does not explicitly identify its article title, leaving some context ambiguity.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:39)

## 40. dev:8839 — FactCheck-GPT — FP

Cluster: **Unstated details or incomplete evidence**. Assessment: `likely_model_error`.

Claim: ECharts Java provides an interface for creating graphs.

Scores: 0.77 / 0.38 / 0.90; combined **0.38**.

A visualization library and image-generation tool do not by themselves establish the claimed graph-creation interface. The checker appears to accept a plausible capability inference.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:40)

## 41. dev:13542 — Lfqa — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Alcohol is also a depressant, which can make people feel more relaxed and sociable, leading to increased desire to drink more.

Scores: 0.68 / 0.09 / 0.90; combined **0.09**.

Relaxation, sociability and further drinking are supported, but the document never classifies alcohol as a depressant. That additional clause can explain rejection of a gold-positive claim.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:41)

## 42. dev:14582 — Lfqa — FN

Cluster: **Strong apparent label/evidence conflicts**. Assessment: `label_review`.

Claim: Artificial sweeteners are not metabolized by the body and thus have no caloric content.

Scores: 0.10 / 0.06 / 0.04; combined **0.04**.

Gold says supported for all artificial sweeteners being un-metabolized and calorie-free, but the reference explicitly states that aspartame is metabolized and yields calories.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:42)

## 43. dev:13140 — Lfqa — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `likely_model_error`.

Claim: Additionally, a clean air filter can also reduce harmful emissions from the engine.

Scores: 0.50 / 0.19 / 0.89; combined **0.19**.

The source says dirty filters cause excess black smoke and soot. Inferring that clean filters can reduce those emissions is reasonable, but the unsupported-detail question vetoes it.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:43)

## 44. dev:14756 — Lfqa — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Throughout its history, Germany has been occupied by various tribes, such as the Alemanni, Saxons, and Germanic people.

Scores: 0.47 / 0.17 / 0.68; combined **0.17**.

Regional tribes and historical habitation are supported, but occupied and throughout its history can imply stronger or broader claims. The three-question rule rejects that historical compression.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:44)

## 45. dev:13959 — Lfqa — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `ambiguous`.

Claim: When adjusting the shutter speed and aperture, it is important to consider the light conditions.

Scores: 0.74 / 0.39 / 0.93; combined **0.39**.

Both controls regulate admitted light, so considering light conditions is a reasonable recommendation. The source does not explicitly issue that advice, leaving the entailment standard uncertain.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:45)

## 46. dev:14967 — Lfqa — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: This method was not very accurate, and was often inaccurate over long distances.

Scores: 0.74 / 0.35 / 0.69; combined **0.35**.

The reference mixes piloting, dead reckoning and map projections. This method could denote different ones; matching an inaccuracy passage without resolving the antecedent can accept the wrong relation.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:46)

## 47. dev:13966 — Lfqa — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `likely_model_error`.

Claim: Reddit can give up loading pages without even trying for a variety of reasons.

Scores: 0.78 / 0.41 / 0.84; combined **0.41**.

The source discusses slowness, outages and crashes; the claim asserts giving up without even trying, a stronger description of internal behavior than the evidence establishes.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:47)

## 48. dev:14971 — Lfqa — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: Another method that was used to map out coasts was known as piloting.

Scores: 0.94 / 0.77 / 0.94; combined **0.77**.

The piloting passage expressly describes using bearings to locate coastline points. The accepted paraphrase looks supported despite the negative label, with some historical-context uncertainty.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:48)

## 49. dev:22600 — RAGTruth — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Overall, Area 5.1 Winery provides a relaxed and enjoyable atmosphere for wine enthusiasts looking for a smaller, more personalized experience.

Scores: 0.50 / 0.11 / 0.68; combined **0.11**.

Mixed tasting reviews and venue attributes are turned into a relaxed, enjoyable, personalized experience. That subjective conclusion exceeds what some reviewers say, despite a supported gold label.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:49)

## 50. dev:24018 — RAGTruth — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `evidence_gap`.

Claim: Gain relevant education and experience: Passage 2 suggests that having a double major in Finance and International Business can be beneficial for a career in investment banking.

Scores: 0.48 / 0.16 / 0.84; combined **0.16**.

A questioner says they have the double major; the passage does not recommend it as beneficial. The gold-positive claim converts a personal circumstance into career advice.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:50)

## 51. dev:27526 — RAGTruth — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: The French-inspired bakery and restaurant serves a variety of delicious pastries, sandwiches, salads, and desserts.

Scores: 0.82 / 0.17 / 0.80; combined **0.17**.

The categories establish French bakery/restaurant offerings, but delicious generalizes across sharply mixed reviews. The broad positive evaluation may trigger the detail veto.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:51)

## 52. dev:26171 — RAGTruth — FN

Cluster: **Strong apparent label/evidence conflicts**. Assessment: `label_review`.

Claim: Focus: QA focuses on defining procedures, rules, and policies that govern the selection and use of raw materials for the manufacturing process, while QC focuses on initiating the evaluation process to ensure the stipulated business goals are realized within the set period of time.

Scores: 0.36 / 0.37 / 0.14; combined **0.14**.

Gold says supported, but the claim attaches ensuring timely business goals to QC; the source assigns that responsibility to QA. The contradiction check rejects the swapped relation.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:52)

## 53. dev:19824 — RAGTruth — FP

Cluster: **Numbers, units and precision**. Assessment: `likely_model_error`.

Claim: The store operates from 10:00 am to 4:00 pm from Monday to Sunday.

Scores: 0.93 / 0.61 / 0.86; combined **0.61**.

The stored hours end at 4:0 in a 24-hour-style range, whereas the claim says 4 pm. The model appears to normalize the unusual overnight schedule into a familiar daytime one.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:53)

## 54. dev:24154 — RAGTruth — FP

Cluster: **Numbers, units and precision**. Assessment: `ambiguous`.

Claim: A North Pacific gray whale named Varvara has set the record for the longest migration of a mammal, swimming 14,000 miles from Russia's Sakhalin Island to Baja, Mexico.

Scores: 0.94 / 0.69 / 0.86; combined **0.69**.

The source says nearly 14,000 miles and describes the route; the claim states 14,000 miles without qualification. Precision or route scope may explain the negative label, but the difference is subtle.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:54)

## 55. dev:20272 — RAGTruth — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: She has been critical of the show and her former co-hosts in the media, and recently, her friend Sunny Hostin suggested she would be perfect for a reality show like The Real Housewives of Potomac.

Scores: 0.90 / 0.53 / 0.74; combined **0.53**.

The source says Sunny considered Meghan a friend, alongside animosity and hostile remarks. Calling Sunny her friend loses the direction and temporal qualification of the relationship.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:55)

## 56. dev:19834 — RAGTruth — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `ambiguous`.

Claim: The exterior of the restaurant is reminiscent of a traditional Italian laneway, and the inside has a cozy, family-friendly atmosphere.

Scores: 0.86 / 0.30 / 0.87; combined **0.30**.

The reviews explicitly mention Italian laneway seating and a quaint family feel. The claim adds interior/cozy wording; the negative label is not clearly explained. It also sits on the floating-point boundary.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:56)

## 57. dev:5111 — Reveal — FN

Cluster: **Numbers, units and precision**. Assessment: `ambiguous`.

Claim: The molecular weight of water is 18 g/mol.

Scores: 0.33 / 0.17 / 0.40; combined **0.17**.

The reference distinguishes molar mass in g/mol from molecular mass in Da and gives 18.0153. The claim uses molecular weight and rounds to 18; technical precision and accepted shorthand conflict.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:57)

## 58. dev:6237 — Reveal — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `ambiguous`.

Claim: Rahul Dravid is a cricketer, not an American football player.

Scores: 0.93 / 0.24 / 0.94; combined **0.24**.

Being a cricketer is explicit; not being an American football player is not. The checker demands evidence for the negative clause that the dataset apparently accepts as an ordinary contrast.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:58)

## 59. dev:5733 — Reveal — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: New Zealand achieved self-government in 1852.

Scores: 0.21 / 0.16 / 0.32; combined **0.16**.

The source distinguishes the 1852 Act from the assembly meeting in 1854 and later limits on government. Achieved self-government compresses several constitutional stages into one date.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:59)

## 60. dev:5413 — Reveal — FN

Cluster: **Numbers, units and precision**. Assessment: `likely_model_error`.

Claim: The land area of the earth is 148,940,000 square kilometers.

Scores: 0.72 / 0.21 / 0.60; combined **0.21**.

The reference's approximate 148,939,063.133 square kilometers rounds to the claim's 148,940,000. The rejection is consistent with excessive sensitivity to numerical precision.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:60)

## 61. dev:5103 — Reveal — FP

Cluster: **Numbers, units and precision**. Assessment: `ambiguous`.

Claim: The oceans contain 1.35e21 liters of water.

Scores: 0.68 / 0.32 / 0.71; combined **0.32**.

The claim's liters closely match a mass-based conversion, whereas the document separately gives a volume of 1.332 billion cubic kilometers. Density assumptions and approximation are not made explicit.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:61)

## 62. dev:6271 — Reveal — FP

Cluster: **Numbers, units and precision**. Assessment: `likely_model_error`.

Claim: The temperature of the outer core of the Earth is about 4,000-7,000 degrees Celsius.

Scores: 0.58 / 0.39 / 0.65; combined **0.39**.

The reference distinguishes outer-region and near-inner-core ranges and gives both Kelvin and Celsius. The claim collapses these into a single different Celsius interval.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:62)

## 63. dev:6320 — Reveal — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: Fox was a Quaker.

Scores: 0.98 / 0.93 / 0.97; combined **0.93**.

Charles Fox's Quaker affiliation is supported, but Fox alone leaves the intended person unspecified. Without the original question, a label disagreement may reflect missing identity context.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:63)

## 64. dev:6115 — Reveal — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: Scoring in the third period is part of hockey.

Scores: 0.92 / 0.52 / 0.95; combined **0.52**.

The reference describes a third-period goal in an Olympic hockey game. The accepted generic claim appears directly supported and the negative label deserves review.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:64)

## 65. dev:1237 — TofuEval-MediaS — FN

Cluster: **Numbers, units and precision**. Assessment: `likely_model_error`.

Claim: However, the new helicopters are over budget at $400 million each and behind schedule.

Scores: 0.78 / 0.23 / 0.73; combined **0.23**.

Over-budget and overdue are explicit, and a later comparison gives around $400 million per helicopter. Combining those statements should support the summary with normal rounding tolerance.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:65)

## 66. dev:2269 — TofuEval-MediaS — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Retiree Bill Sand designs solutions enabling a cello-playing sister duo and a young girl with limited mobility to pursue their passions.

Scores: 0.87 / 0.22 / 0.71; combined **0.22**.

The devices, children and age 66 are established, but the source never calls Bill Sand retired. One extra status word can veto an otherwise faithful summary.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:66)

## 67. dev:1963 — TofuEval-MediaS — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `evidence_gap`.

Claim: Former North Korean leader Kim Jong-il, who controls the country's legal system, may influence the outcome of the trial.

Scores: 0.90 / 0.27 / 0.82; combined **0.27**.

The historical broadcast calls Kim Jong-il the country's leader; the summary calls him former and adds control of the legal system. Those additions are not established in the supplied text.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:67)

## 68. dev:2247 — TofuEval-MediaS — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Bill Sand builds custom scooters, cello stands for armless sisters and other tools allowing a girl with Miller's Syndrome to write and be mobile.

Scores: 0.83 / 0.28 / 0.72; combined **0.28**.

The scooter and cello stands already exist, while the writing aid is only being brainstormed. The summary compresses planned and completed assistance into a single accomplishment.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:68)

## 69. dev:1550 — TofuEval-MediaS — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: If Scotland votes to leave UK, its biggest challenge will be currency.

Scores: 0.84 / 0.30 / 0.88; combined **0.30**.

An economist calls currency the biggest unknown; the claim calls it the biggest challenge as a general fact. This loses attribution and subtly strengthens the claim; a boundary artifact also allows acceptance.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:69)

## 70. dev:1264 — TofuEval-MediaS — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: The new helicopter is designed to protect the president from missile attacks and improve communications, but its price tag is higher than the cost of Air Force One.

Scores: 0.90 / 0.71 / 0.87; combined **0.71**.

The source explicitly confirms the comparison, giving $400 million versus $325 million per Air Force One plane. The negative label may reflect the earlier conflicting phrase as much as.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:70)

## 71. dev:2010 — TofuEval-MediaS — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `likely_model_error`.

Claim: Democrats said the focus should be on the Canadian border.

Scores: 0.87 / 0.50 / 0.79; combined **0.50**.

Pointing to vulnerabilities at the Canadian border becomes an explicit prescription to focus there. The model accepts the inferred policy position although the source reports a narrower statement.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:71)

## 72. dev:2624 — TofuEval-MediaS — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `likely_model_error`.

Claim: The Palestinian ambassador criticized the Trump administration's peace plan and decision to move the U.S. embassy to Jerusalem.

Scores: 0.92 / 0.44 / 0.81; combined **0.44**.

The interview criticizes the administration's actions and peace process, while saying its specific plan is not yet released. The summary treats criticism as directed at the plan itself.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:72)

## 73. dev:3858 — TofuEval-MeetB — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `likely_model_error`.

Claim: The document discusses a property development project by Raintree in Long Beach.

Scores: 0.87 / 0.29 / 0.83; combined **0.29**.

Raintree's project, council discussion and Long Beach are distributed across a noisy transcript. The direct-support score is high but the detail question fails to accept the combined location/topic summary.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:73)

## 74. dev:3836 — TofuEval-MeetB — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: Without records, citizens cannot file complaints or get information to protect their rights.

Scores: 0.82 / 0.21 / 0.85; combined **0.21**.

A speaker's complaint about inability to obtain records becomes an unqualified statement that citizens cannot file complaints or protect rights. It may be faithful summarization or overgeneralization.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:74)

## 75. dev:3181 — TofuEval-MeetB — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: The city has also produced numerous jazz and blues musicians, many of whom have been influential in shaping the genre.

Scores: 0.83 / 0.15 / 0.91; combined **0.15**.

The transcript names celebrated Seattle musicians and their accomplishments; influential in shaping the genre is a broader historical assessment than the specific examples.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:75)

## 76. dev:3840 — TofuEval-MeetB — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `likely_model_error`.

Claim: The City Council discussed property development by Raintree for two projects in downtown Long Beach.

Scores: 0.90 / 0.28 / 0.85; combined **0.28**.

The transcript explicitly mentions two downtown projects, Raintree and Long Beach in different turns. Fragmented evidence and transcription noise plausibly explain the unsupported-detail veto.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:76)

## 77. dev:3790 — TofuEval-MeetB — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: Tesoro recently increased gas prices $0.30/gallon.

Scores: 0.89 / 0.42 / 0.87; combined **0.42**.

The transcript says fuel prices rose after a Tesoro strike. The claim changes that into Tesoro itself raising its prices, switching a causal background event into an actor's decision.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:77)

## 78. dev:4543 — TofuEval-MeetB — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `likely_model_error`.

Claim: The recommendation is to approve an implementation term sheet for the Collaborative Building Futures project, which aims to construct new supportive housing facilities for women and children on a 10.4 acre parcel in Alameda Point.

Scores: 0.83 / 0.35 / 0.78; combined **0.35**.

The agenda lists Collaborative Building Futures with Women and Children among several organizations. The claim treats that name as the project and target residents, losing entity boundaries.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:78)

## 79. dev:3588 — TofuEval-MeetB — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: The Police Department is requesting authorization to receive and spend grant funding of up to $368,000 for the implementation of a body-worn camera policy.

Scores: 0.98 / 0.92 / 0.96; combined **0.92**.

The police department's report requests authorization for the city manager to receive and spend the grant. The summary may misassign the authorized actor, despite matching the amount and purpose.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:79)

## 80. dev:4590 — TofuEval-MeetB — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `likely_model_error`.

Claim: The document discusses the budget performance for the fiscal year 2016 citywide.

Scores: 0.78 / 0.77 / 0.86; combined **0.77**.

The budget report appears only in the final announcement of the next agenda item. The model treats a topic mention as evidence that the document discusses that topic.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:80)

## 81. dev:4932 — Wice — FN

Cluster: **Unstated details or incomplete evidence**. Assessment: `ambiguous`.

Claim: On October 6, 1995, the band released the studio version of the song to radio via satellite uplink to stem excessive spread of taped copies of the song.

Scores: 0.90 / 0.29 / 0.78; combined **0.29**.

Radio release, date and prevention of taped copies are supported. Studio version is implicit from the album context, so literal detail checking may reject an acceptable inference.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:81)

## 82. dev:4813 — Wice — FN

Cluster: **Paraphrase and combining evidence**. Assessment: `likely_model_error`.

Claim: On December 20, 2018, Tom Perez, the chairman for the Democratic National Committee, announced the preliminary schedule for a series of official debates, set to begin in June 2019.

Scores: 0.85 / 0.22 / 0.61; combined **0.22**.

The publication date, DNC announcement, Perez quotation and next June jointly establish the claim. The checker appears reluctant to combine metadata and temporal context across the article.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:82)

## 83. dev:4970 — Wice — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: The collapse has been seen in other alt-right groups, and has been attributed to widespread public backlash against neo-Nazism and white supremacy since the 2017 Charlottesville rally.

Scores: 0.78 / 0.19 / 0.79; combined **0.19**.

The source reports backlash and movement fraying but explicitly says the exact reason is unclear. The claim supplies a broader causal attribution; the gold-positive label tolerates that compression.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:83)

## 84. dev:4741 — Wice — FN

Cluster: **Scope, qualifiers and event status**. Assessment: `evidence_gap`.

Claim: Young faced Kedzie again in a five-round title rematch at Jackson's MMA Series 4 on April 9, 2011 in Albuquerque, New Mexico.

Scores: 0.15 / 0.08 / 0.21; combined **0.08**.

The source announces a future fight and says earlier show results are unreleased. The claim says the fight happened and was a rematch. Those event-status and recurrence details are not established.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:84)

## 85. dev:4987 — Wice — FP

Cluster: **Unstated details or incomplete evidence**. Assessment: `ambiguous`.

Claim: During the Hyde Park concert, British comedian Peter Kay jokingly introduced the Spice Girls while he was introducing The Who.

Scores: 0.92 / 0.70 / 0.77; combined **0.70**.

Peter Kay's joke and introduction are explicit, but British nationality is not stated. The negative label might penalize that extra attribute; the event itself is well supported.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:85)

## 86. dev:4941 — Wice — FP

Cluster: **Entities, attribution and unresolved references**. Assessment: `ambiguous`.

Claim: This became an impetus for galleries and museums across the UK to celebrate "the making, debating and exhibiting art at the Royal Academy".

Scores: 0.83 / 0.49 / 0.89; combined **0.49**.

The anniversary and countrywide celebrations are explicit, but this has no resolved antecedent. A causal claim about what motivated the celebrations may depend on omitted prior context.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:86)

## 87. dev:4683 — Wice — FP

Cluster: **Scope, qualifiers and event status**. Assessment: `ambiguous`.

Claim: A second Industry Summit was held in 2018, and plans are to continue this again in November 2019.

Scores: 0.90 / 0.47 / 0.86; combined **0.47**.

The 2019 date and links to 2017/2018 summits support recurrence, but second in 2018 requires assuming the list is exhaustive. That ordinal is not explicitly established.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:87)

## 88. dev:4937 — Wice — FP

Cluster: **Paraphrase and combining evidence**. Assessment: `label_review`.

Claim: Following criticism against her for choosing to work with him, Newton was replaced by Tate Taylor.

Scores: 0.97 / 0.85 / 0.92; combined **0.85**.

One source reports criticism of Chastain; the next reports Newton's withdrawal and replacement by Tate Taylor. The claim appears supported by combining the sources, though following can carry a causal implication.

[Full validation record](/home/jacopodardini/work/jev-check/results/validation_error_analysis/review_sample.jsonl:88)
