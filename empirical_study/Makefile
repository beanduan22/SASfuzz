PY := PYTHONPATH=src python3 -m

.PHONY: all collect hydrate verify classify report clean confirm

all: report

collect:
	$(PY) empirical_study.collect

hydrate:
	$(PY) empirical_study.hydrate

verify:
	$(PY) empirical_study.verify_fix --deep

classify:
	$(PY) empirical_study.classify --backend heuristic

report:
	$(PY) empirical_study.report

confirm: collect report
	@echo "--- empirical_study_report.md ---"
	@cat reports/empirical_study_report.md

clean:
	rm -rf data/raw_search.json data/filtered_issues.json data/collect_summary.json \
	       data/filtered_issues_full.json data/hydrate_summary.json \
	       data/fix_verification.json data/fix_verification_summary.json \
	       data/classification.json data/classification_summary.json \
	       reports/empirical_study_report.md reports/empirical_study_numbers.json
