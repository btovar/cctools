#!/bin/sh

. ../../dttools/test/test_runner_common.sh

# This is a true unit test: vine_task_unit_test.py exercises the pure-Python
# logic of ndcctools.taskvine.task.Task directly (input validation, the
# inputs/outputs and features normalization in __init__, and the deferred
# output-registration bookkeeping), using a hand-written stand-in for the
# SWIG-generated cvine C extension. Like TR_vine_manager_test.sh, it needs
# neither a running manager process nor run_taskvine_worker.
#
# Unlike the other TR_vine_python_* tests, this does not use
# CCTOOLS_PYTHON_TEST_EXEC / test_support's installed python_modules: it
# only needs python3's stdlib (unittest, unittest.mock) plus the taskvine
# python bindings source tree itself (added to PYTHONPATH below), so it
# runs even when the taskvine python bindings have not been built (no
# swig available).

PYTHON_TEST_EXEC=${CCTOOLS_PYTHON_TEST_EXEC:-python3}

export PYTHONPATH=$(pwd)/../src/bindings/python3:$PYTHONPATH

check_needed()
{
	command -v "${PYTHON_TEST_EXEC}" > /dev/null 2>&1
}

prepare()
{
	return 0
}

run()
{
	"${PYTHON_TEST_EXEC}" vine_task_unit_test.py
	return $?
}

clean()
{
	return 0
}

dispatch "$@"

# vim: set noexpandtab tabstop=4:
