# simple makefile to simplify repetetive build env management tasks under posix
# caution: testing won't work on windows, see README
PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.html,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat"
CODESPELL_DIRS ?= meegkit/ doc/

help:
	@echo "Please use \`make <target>' where <target> is one of:"
	@echo "  all		to build everything"
	@echo "  init		to check all external links for integrity"
	@echo "  clean		to run all doctests embedded in the documentation (if enabled)"
	@echo "  build-doc	to make standalone HTML documentation files"
	@echo "  pep		to run PEP8 checks"
	@echo "  pydocstyle	to check docstyle"
	@echo "  flake		to run flake8"
	@echo "  test		to run tests"

all: clean inplace test test-doc

# pip install -r requirements.txt
init:
	conda env create -f environment.yml

# Cleaning
# =============================================================================
clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-build:
	rm -rf _build

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-cache

# Doc
# =============================================================================
build-doc:
	cd doc; make clean
	cd doc; make html

build-examples:
	cd examples;
	find . -name "example_*.py" | xargs sphx_glr_python_to_jupyter.py
	find . -name "example_*.ipynb" | xargs jupyter nbconvert --execute --to notebook --inplace

# Style
# =============================================================================
codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

pep:
	@$(MAKE) -k flake pydocstyle codespell-error

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count meegkit examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

# Tests
# =============================================================================
# test:
# 	py.test tests

test: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' meegkit

test-verbose: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' meegkit --verbose

test-fast: in
	rm -f .coverage
	$(PYTESTS) -m 'not slowtest' meegkit

test-full: in
	rm -f .coverage
	$(PYTESTS) meegkit

.PHONY: init test
