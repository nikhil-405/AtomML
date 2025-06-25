set -e
coverage run --source=. -m pytest
coverage report -m
coverage xml -i