.PHONY: all test test_a test_b results_a results_b diff_results clean

all: results_a results_b

test: test_a test_b

test_a:
	python -m pytest project_a/tests/ -v

test_b:
	python -m pytest project_b/tests/ -v

results_a: test_a
	python -m project_a.measure --llm-limit 10 > RESULTS_A_generated.md
	@echo "Generated RESULTS_A_generated.md. Diff against RESULTS_A.md to verify reproducibility."

results_b: test_b
	python -m project_b.measure --sample-size 1000 --llm-sample-size 20 --rate 0.05 > RESULTS_B_generated.md
	@echo "Generated RESULTS_B_generated.md. Diff against RESULTS_B.md to verify reproducibility."

diff_results:
	@echo "=== Diffing Project A Results ==="
	@diff -u RESULTS_A_generated.md RESULTS_A.md || true
	@echo "=== Diffing Project B Results ==="
	@diff -u RESULTS_B_generated.md RESULTS_B.md || true

clean:
	rm -f RESULTS_A_generated.md RESULTS_B_generated.md
