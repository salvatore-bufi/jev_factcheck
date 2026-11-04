"""Dataset builders for the evidence head: every source becomes the same model-independent row format.

    python -m evidence.data.build --source scifact wice hover      # -> dataset/evidence/<source>.jsonl
    python -m evidence.data.show --source hover --n 5               # read rows with their evidence marked

Format and rules: evidence/data/schema.py; plan: Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/02_dati_fonti.md;
docs: Documentation/fine_tuning_factcheck_evidence/testa_per_frase/02_dati_fonti.md.
"""
