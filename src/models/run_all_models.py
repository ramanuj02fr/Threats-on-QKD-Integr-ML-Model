"""Run models, combine their results, and create all Matplotlib figures."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["model_logistic_regression.py", "model_random_forest.py", "model_knn.py",
           "model_gradient_boosting.py", "model_svm.py", "model_neural_network.py"]

for s in SCRIPTS:
    print(f"\n########## {s} ##########", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)] + sys.argv[1:])
    if r.returncode != 0:
        sys.exit(f"{s} failed (exit code {r.returncode})")

if len(sys.argv) == 1:
    for script in ("combine_model_results.py", "make_figures.py"):
        print(f"\n########## {script} ##########", flush=True)
        result = subprocess.run([sys.executable, os.path.join(HERE, script)])
        if result.returncode != 0:
            sys.exit(f"{script} failed (exit code {result.returncode})")
    print("\nComplete: tables are in results/combined and figures are in results/figures.")
else:
    print("\nPartial run: tables and figures were skipped because extra arguments were supplied.")
