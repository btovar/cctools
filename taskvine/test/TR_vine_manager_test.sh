#!/bin/sh

. ../../dttools/test/test_runner_common.sh

# This is a true unit test: vine_manager_test calls a handful of
# vine_manager.c functions directly with hand-constructed inputs. It does
# not create a manager or a worker, so unlike most other TR_vine_* tests it
# needs neither a running manager process nor run_taskvine_worker.

prepare()
{
	return 0
}

run()
{
	../src/tools/vine_manager_test
	return $?
}

clean()
{
	return 0
}

dispatch "$@"

# vim: set noexpandtab tabstop=4:
