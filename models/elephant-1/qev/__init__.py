"""Qev, the serving runtime of elephant-1: a copy of six modules of OneJev's `qev` package (Apache-2.0, LICENSE here).

Source: https://github.com/OmniJev/OneJev at 81ce62f1597c91e46767d4d02d6ac2e18534fe94 with the checkpoint's own
serve.patch applied (baselabs/elephant-1 revision b3f8a93), which is what the model card's setup.sh installs.
engine.py, readout.py, prompt.py, schema.py, answers.py and calibrate.py are byte-identical to that source. This file
replaces the package's __init__.py, which only re-exported the HTTP client (qev.client, not needed here).
"""
__version__ = "0.1.0"
