set -e
coverage run --source=ml_scratch -m pytest
coverage report -m
coverage xml -i