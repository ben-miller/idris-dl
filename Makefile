.PHONY: watch test

test:
	idris2 --exec main Test.idr

watch:
	watchexec -e idr -- idris2 -p test --exec main Test.idr
