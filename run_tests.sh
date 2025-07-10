#!/bin/bash
pytest -v -p no:warnings slynx_test.py > logs/test_results_$(date +"%Y-%m-%d_%H-%M-%S").txt
echo "Test results saved to logs/"
