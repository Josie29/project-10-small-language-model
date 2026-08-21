# Mislocalisation audit - v1 checkpoints

Every v1 trial the judge failed with `wrong_lifetime_focus`, with the judge's own
`has_localization` check beside it. The label reads as *cannot find the buggy line*.
The transcripts say the line was found in **23 of 24** cases - what failed is
the question asked about it.

Source: `results/base-vs-tuned/trials.jsonl` (eval commit `3b637d1`). Only rows where `model_id` is a tuned v1
checkpoint; the 28 v1 failures break down as 24 `wrong_lifetime_focus`, 3 `emitted_code`, 1 `no_localization`.

Regenerate:

```bash
jq -c 'select(.model_id|startswith("machalek29")) | select(.verdict.violation=="wrong_lifetime_focus")' \
  results/base-vs-tuned/trials.jsonl
```

## The mismatch, three rows

| bug concept | what the model asked instead |
| --- | --- |
| reset | "When does that set get **created**?" (`clean-15`, n-125) |
| aliasing | "Which list does that list **create**?" (`clean-24`, n-125) |
| ownership | "When does that list get **emptied**?" (`adv-05`, n-125) |

## All 24 trials

| N | scenario | bug concept | quoted the bug region | question asked |
| ---: | --- | --- | :---: | --- |
| 62 | `clean-02` | creation | yes | Look at `counts={}`. When does that dictionary get populated? |
| 62 | `clean-05` | creation | yes | Consider `log=[]` inside the function. When does that list get appended? |
| 62 | `clean-09` | ownership | yes | Look at `recent_searches = []` in the code block. When does that list get populated? |
| 62 | `clean-11` | ownership | yes | Look at `notifications = []` in the code block. When does that list get created? |
| 62 | `clean-12` | ownership | yes | Look at `members = set()` in the class body. When does that set get populated? |
| 62 | `clean-13` | reset | yes | Look at `names = []` in the loop body. When does that list get extended? |
| 62 | `clean-15` | reset | yes | Consider `words = set()` in the loop body. When does that set get updated? |
| 62 | `clean-17` | reset | yes | Look at `all_rows = []` inside the loop. When does that list get extended? |
| 62 | `clean-19` | aliasing | yes | Look at `backup = original`. When does that list go out of scope? |
| 62 | `clean-20` | aliasing | **no** | Look at `defaults = {"theme": "light"}`. When does that dictionary get created? |
| 62 | `clean-21` | aliasing | yes | Look at `grid = [[0] * 3] * 3` first. Which list comprehension creates the grid? |
| 62 | `clean-23` | aliasing | yes | Look at `pending = ["send", "archive"]` in the code block. When does that list get assigned to `active = pending`? |
| 62 | `clean-24` | aliasing | yes | Look at `buckets = [[]] * 3` first. Which list is that? |
| 62 | `adv-10` | aliasing | yes | Look at `copy_for_report = source`. When does that list start copying? |
| 62 | `adv-11` | aliasing | yes | Look at `seats = [[None] * 4] * 2` first. Which list is that? |
| 62 | `adv-12` | aliasing | yes | Look at `template = {"enabled": False}`. When does that dictionary get copied into `feature = template`? |
| 125 | `clean-10` | ownership | yes | Look at `flags = {}` in the class body. Which Session objects will overwrite that dictionary? |
| 125 | `clean-11` | ownership | yes | Notice `notifications = []` at the top. When does that list get created? |
| 125 | `clean-15` | reset | yes | Look at `words = set()` in the signature. When does that set get created? |
| 125 | `clean-21` | aliasing | yes | Look at `grid = [[0] * 3] * 3`. Which list does that create? |
| 125 | `clean-24` | aliasing | yes | Look at `buckets = [[]] * 3`. Which list does that list create? |
| 125 | `adv-05` | ownership | yes | Look at `audit_log = []` at the top. When does that list get emptied? |
| 125 | `adv-09` | reset | yes | Look at `ids = set()` in the signature. When does that set get created? |
| 125 | `adv-11` | aliasing | yes | Focus on `seats = [[None] * 4] * 2`. Which list does that list build? |

n-250 and n-500 contribute zero rows: both score 100% on this eval set. That is the
number the shape-swap probe was built to test.
