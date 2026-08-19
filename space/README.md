---
title: Python State-Lifetime Tutor
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

Base vs tuned comparison for a Qwen3-0.6B fine-tuned to hold one teaching behavior:
localize a Python mutable-state lifetime bug and ask exactly one non-compound question
about it, never stating the fix.

The base model is given the full behavior spec as a system prompt. The tuned model is
given one line. Both run on CPU.

## Deploying

Hosted on Railway at
<https://slm-state-lifetime-demo-production.up.railway.app>. `railway.json` pins the
Dockerfile builder, but **one setting lives only in Railway** and is not reproducible from
this repo: the service's *root directory* must be set to `space`. `railway up` uploads from
the linked project path, which is the repository root, so without it Railway autodetects
the Python project at the root and never sees this Dockerfile.

Requirement ranges here are capped on purpose - see the note in `requirements.txt`.
