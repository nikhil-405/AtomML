set -e
coverage run --source=AtomML -m pytest
coverage report -m
coverage xml -i