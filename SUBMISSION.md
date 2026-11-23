# Submission checklist — GraphOne / FrontierAtlas demo task

Status: everything that can be done before submission day is done.

## Done
- [x] Phase I full run: 1,000 startups / 1,000 products / 1,000 papers
      (990 with live GitHub stars) — `data/phase1_report.json`
- [x] Phase IV entity resolution on the full set: 3 seed, 1 alias,
      7 review, 989 new — `data/canonical/entity_mapping_log.jsonl`
- [x] LangGraph pipeline (`src/graph.py`), FastAPI (`src/api.py`),
      OpenRouter primary LLM tier, DeepEval-style eval harness
- [x] Architecture doc + PDF, exactly 3 pages — `architecture.pdf`
- [x] LiteLLM gateway + guardrails + DeepEval suite (official DeepEval
      4.3.4 metrics; live run 23 Nov 2026: all metrics >= 0.93, Field F1 1.00)
- [x] 6-tab export ready — `data/sheets/*.csv`
- [x] This repo committed locally (see "Push" below)

## On submission day (do in this order)
1. Run the fresh news/jobs crawl (the 24h window goes stale):
   `.venv/bin/python -m src.phase2 --full`
2. Re-export the tabs: `.venv/bin/python -m src.export_sheets`
3. Upload to Google Sheets:
   `GOOGLE_SHEETS_CREDENTIALS=/path/to/sa.json .venv/bin/python -m src.export_sheets --upload`
   (needs a Google service-account JSON; without it, import the 6 CSVs
   manually into a Sheet and share it publicly)
4. Push this repo to GitHub (see below) and confirm the README renders.
5. Submit the form: https://forms.gle/3JJhukomumTFueRW7
   with the public Sheet URL and the GitHub repo URL.

## Push
```bash
# create the repo on github.com/Rajendra-Pd-Joshi first (web UI or `gh repo create`)
git remote add origin git@github.com:Rajendra-Pd-Joshi/frontieratlas.git
git push -u origin main
```

## Still open (needs the user)
- OpenRouter API key, for a live LLM extraction run
  (`.venv/bin/python -m src.eval_extraction --live`)
- Exact deadline (assumed ~26 Nov 2026, unconfirmed)
- Product Hunt developer token (optional: pricing diversity for Products)
