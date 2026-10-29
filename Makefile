.PHONY: install validate-resources clean

install:
	python -m pip install -e .

# Checks the packaged Scenario 2 part tags (42 subclasses, 813 images). Pass
# --mvtec-root / --apd-root to `tgad validate` to also check dataset copies.
validate-resources:
	tgad validate

clean:
	rm -rf build dist *.egg-info src/*.egg-info
