SPHINXBUILD ?= sphinx-build
SPHINXOPTS ?= -W --keep-going
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help Makefile
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
