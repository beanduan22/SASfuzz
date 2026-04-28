PY := PYTHONPATH=src python3 -m

.PHONY: all collect hydrate verify classify report clean confirm

all: report

collect:
	$(PY) rq1.collect

hydrate:
	$(PY) rq1.hydrate

verify:
	$(PY) rq1.verify_fix --deep

classify:
	$(PY) rq1.classify --backend heuristic

report:
	$(PY) rq1.report

confirm: collect report
	@echo "--- RQ1_report.md ---"
	@cat reports/RQ1_report.md

clean:
	rm -rf data/raw_search.json data/filtered_issues.json data/collect_summary.json \
	       data/filtered_issues_full.json data/hydrate_summary.json \
	       data/fix_verification.json data/fix_verification_summary.json \
	       data/classification.json data/classification_summary.json \
	       reports/RQ1_report.md reports/RQ1_numbers.json
