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
