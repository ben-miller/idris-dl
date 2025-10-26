.PHONY: watch test

test:
	idris2 --exec main Test.idr

watch:
	watchexec -e idr -- idris2 -p test --exec main Test.idr

test_mnist: 
	idris2 --exec main TestMNIST.idr

watch_mnist:
	watchexec -e idr -- idris2 -p test --exec main TestMNIST.idr

