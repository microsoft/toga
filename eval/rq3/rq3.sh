python gen_tests_from_metadata.py ../../data/oracle_preds.csv ../../data/evosuite_regression_all
python aggregate_test_cases.py toga_generated/
python eval_tests.py toga_generated/aggregated_d4j_tests/ -o toga_generated/results/
python collect_test_results.py toga_generated/
