#!/usr/bin/env python3

# Copyright (C) 2026- The University of Notre Dame
# This software is distributed under the GNU General Public License.
# See the file COPYING for details.

"""
Unit tests for the pure-Python logic in
ndcctools.taskvine.task.Task (taskvine/src/bindings/python3/ndcctools/taskvine/task.py).

This is a *true* unit test: it does not start a taskvine manager or
worker, and it does not require the real, SWIG-generated `cvine` C
extension to be built. It targets a handful of Task.__init__ /
Task.add_input / Task.add_output code paths that contain real branching
and validation logic, as opposed to trivial getters/setters that just
forward a value to a C call.

Why a fake `cvine` is needed
-----------------------------
`ndcctools.taskvine.task` imports `from . import cvine`, and
`Task.__init__` immediately calls `cvine.vine_task_create(command)`. The
`cvine` module is the SWIG-generated wrapper around the C taskvine
library; it is not built in this environment (no swig available), and
there is no existing stub for it anywhere in this repo.

Importing `ndcctools.taskvine` the normal way (`import ndcctools.taskvine`)
is not an option either: that runs `ndcctools/taskvine/__init__.py`, which
imports `manager.py`, `futures.py`, `dask_executor.py`, etc. -- all of
which either import the real `cvine` directly or transitively depend on
things that are not available here (and are explicitly out of scope: this
task must not touch manager.py).

So instead we import `ndcctools.taskvine.task` in isolation:
  1. We register a hand-written stand-in module object in
     `sys.modules["ndcctools.taskvine"]` *before* importing the `task`
     submodule. This stand-in has a `__path__` pointing at the real
     `ndcctools/taskvine` directory on disk, but its own `__init__.py`
     code is never executed, since Python's import machinery only runs a
     package's `__init__.py` when the package is not already present in
     `sys.modules`.
  2. We register a minimal fake `cvine` module in
     `sys.modules["ndcctools.taskvine.cvine"]`. When `task.py` (and its
     sibling `file.py` / `utils.py`, which task.py also pulls in via
     `from .file import File` / `from .utils import get_c_constant`) does
     `from . import cvine`, Python resolves that against our fake package
     and finds the stub already sitting in `sys.modules`, so the real
     (unbuilt) extension is never touched.
  3. With those two stand-ins in place, `importlib.import_module` loads
     `task.py` (and `file.py`, `utils.py`) *for real*, straight off disk,
     via the normal import machinery. Every branch of Task.__init__,
     Task.add_input, Task.add_output, Task._determine_mount_flags, etc.
     that we then call is genuine task.py code -- nothing about the
     logic under test is reimplemented in this file. Only the leaf calls
     into the (unbuilt) C extension are replaced by recording
     unittest.mock.MagicMock objects.

Fidelity / limitations of this approach
----------------------------------------
- The fake `cvine.vine_task_add_input` / `vine_task_add_output` /
  `vine_task_add_feature` / etc. calls are recorded but do not perform
  any real C-side validation. So these tests can verify *what arguments
  task.py's Python logic computes and passes down* (e.g. the exact
  bit-flags produced by `_determine_mount_flags`, or the exact
  `remote_name` extracted from a dict-vs-str "inputs"/"outputs" value),
  but they cannot verify that the real C library accepts those arguments
  correctly. That would require the built extension, which is not
  available in this sandbox.
- The bit-flag constants below (VINE_WATCH, VINE_FAILURE_ONLY, ...) are
  given distinct, non-overlapping bit positions specifically so that
  tests can detect if `_determine_mount_flags`/`_determine_file_flags`
  stop actually OR-ing the requested bits together (as opposed to, say,
  always returning a fixed value that happens to satisfy a truthiness
  check).
- `Task.__del__` is exercised incidentally (Python calls it when test
  Task objects are garbage collected); the fake `cvine.vine_task_delete`
  just records the call.
"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

REPO_PY_BINDINGS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "bindings", "python3")
)
if REPO_PY_BINDINGS not in sys.path:
    sys.path.insert(0, REPO_PY_BINDINGS)


def _install_fake_cvine():
    """Build and register a minimal stand-in for the ndcctools.taskvine.cvine
    SWIG extension: just enough surface for Task.__init__/add_input/add_output
    and the small static helper methods to run.
    """
    fake_cvine = types.ModuleType("ndcctools.taskvine.cvine")

    # Bit-flag constants used by Task._determine_mount_flags / _determine_file_flags.
    # Distinct bit positions so tests can confirm the real OR-ing logic in task.py,
    # not just that "some truthy value" came back.
    fake_cvine.VINE_TRANSFER_ALWAYS = 1 << 0
    fake_cvine.VINE_WATCH = 1 << 1
    fake_cvine.VINE_FAILURE_ONLY = 1 << 2
    fake_cvine.VINE_SUCCESS_ONLY = 1 << 3
    fake_cvine.VINE_FIXED_LOCATION = 1 << 4
    fake_cvine.VINE_MOUNT_SYMLINK = 1 << 5
    fake_cvine.VINE_PEER_NOSHARE = 1 << 6
    fake_cvine.VINE_UNLINK_WHEN_DONE = 1 << 7

    # Cache-level constants used by Task._determine_cache_level. Deliberately
    # not 0/1/2/3 in order, so a test that just checks "truthy" or "in range"
    # would not accidentally pass.
    fake_cvine.VINE_CACHE_LEVEL_TASK = 100
    fake_cvine.VINE_CACHE_LEVEL_WORKFLOW = 101
    fake_cvine.VINE_CACHE_LEVEL_WORKER = 102
    fake_cvine.VINE_CACHE_LEVEL_FOREVER = 103

    # Task lifecycle. vine_task_create must return something truthy, or
    # Task.__init__ raises "Unable to create internal Task structure".
    fake_cvine.vine_task_create = mock.MagicMock(return_value="fake-task-handle")
    fake_cvine.vine_task_delete = mock.MagicMock()
    fake_cvine.vine_task_addref = mock.MagicMock(return_value="fake-task-handle-ref")

    # add_input/add_output return 0 to signal failure (task.py raises
    # ValueError in that case); default to "success".
    fake_cvine.vine_task_add_input = mock.MagicMock(return_value=1)
    fake_cvine.vine_task_add_output = mock.MagicMock(return_value=1)
    fake_cvine.vine_task_add_feature = mock.MagicMock()
    fake_cvine.vine_task_set_env_var = mock.MagicMock()

    sys.modules["ndcctools.taskvine.cvine"] = fake_cvine
    return fake_cvine


def _install_fake_cloudpickle_if_missing():
    """task.py does a top-level `import cloudpickle`, used only inside
    PythonTask methods (none of which this file's tests exercise -- those
    methods need a real Manager, which is out of scope here). cloudpickle
    is a third-party dependency that is not installed in this sandbox (no
    network access to pip). If it's genuinely available, use the real
    thing; otherwise install a trivial stub purely so the module-level
    `import cloudpickle` in task.py succeeds. This stub is never invoked
    by any code path under test.
    """
    try:
        import cloudpickle  # noqa: F401
        return
    except ImportError:
        pass

    fake_cloudpickle = types.ModuleType("cloudpickle")
    fake_cloudpickle.dump = mock.MagicMock()
    fake_cloudpickle.dumps = mock.MagicMock()
    fake_cloudpickle.load = mock.MagicMock()
    fake_cloudpickle.loads = mock.MagicMock()
    sys.modules["cloudpickle"] = fake_cloudpickle


def _load_task_module():
    """Import ndcctools.taskvine.task in isolation (see module docstring)."""
    _install_fake_cloudpickle_if_missing()
    import ndcctools  # noqa: F401  (real import: trivial/empty __init__.py)

    taskvine_dir = os.path.join(REPO_PY_BINDINGS, "ndcctools", "taskvine")

    fake_pkg = types.ModuleType("ndcctools.taskvine")
    fake_pkg.__path__ = [taskvine_dir]
    fake_pkg.__package__ = "ndcctools.taskvine"
    sys.modules["ndcctools.taskvine"] = fake_pkg

    fake_cvine = _install_fake_cvine()
    fake_pkg.cvine = fake_cvine

    task_module = importlib.import_module("ndcctools.taskvine.task")
    return task_module, fake_cvine


TASK_MODULE, FAKE_CVINE = _load_task_module()
Task = TASK_MODULE.Task
File = TASK_MODULE.File  # the *real* File class task.py itself imports and checks isinstance() against


def make_file(label):
    """A File wraps an opaque internal handle; task.py never dereferences it
    beyond passing `file._file` straight through to the (faked) C layer."""
    return File(f"internal-file-{label}")


class TestTaskInitCommandAndFeatures(unittest.TestCase):
    """Task.__init__: command-type validation and the features str-vs-list-vs-invalid branch."""

    def setUp(self):
        FAKE_CVINE.vine_task_add_feature.reset_mock()

    def test_command_as_dict_raises_typeerror(self):
        # Task.__init__ has an explicit isinstance(command, dict) guard meant to
        # catch the common mistake of calling Task(some_dict) instead of
        # Task(cmd, **some_dict). This must reject the dict *before* ever
        # touching cvine.vine_task_create.
        with self.assertRaises(TypeError):
            Task({"cores": 1})

    def test_command_as_str_is_accepted(self):
        t = Task("echo hello")
        self.assertIsNotNone(t)

    def test_features_as_single_string_adds_one_feature(self):
        Task("echo hello", features="gpu-node")
        FAKE_CVINE.vine_task_add_feature.assert_called_once()
        args = FAKE_CVINE.vine_task_add_feature.call_args[0]
        self.assertEqual(args[1], "gpu-node")

    def test_features_as_list_adds_each_feature_in_order(self):
        Task("echo hello", features=["gpu-node", "large-disk", "ssd"])
        got = [call.args[1] for call in FAKE_CVINE.vine_task_add_feature.call_args_list]
        self.assertEqual(got, ["gpu-node", "large-disk", "ssd"])

    def test_features_of_invalid_type_raises(self):
        # features must be a str or a list; anything else (e.g. a plain int)
        # should be rejected rather than silently ignored or passed through.
        with self.assertRaises(Exception):
            Task("echo hello", features=42)
        FAKE_CVINE.vine_task_add_feature.assert_not_called()


class TestTaskInputsOutputsNormalization(unittest.TestCase):
    """Task.__init__: the inputs={File: str-or-dict} / outputs={File: str-or-dict}
    normalization logic, which is real branching/validation code (not a
    passthrough)."""

    def setUp(self):
        FAKE_CVINE.vine_task_add_input.reset_mock()
        FAKE_CVINE.vine_task_add_output.reset_mock()

    def test_inputs_str_value_is_normalized_to_remote_name(self):
        f = make_file("in1")
        t = Task("echo hello", inputs={f: "remote_in.txt"})

        # _tracked_inputs is the cache-fingerprinting bookkeeping list
        # mentioned in the file's own comments; verify it recorded exactly
        # what was requested.
        self.assertEqual(t._tracked_inputs, [(f, "remote_in.txt")])

        # str "remote_in.txt" must become add_input(file, remote_name="remote_in.txt")
        # with strict_input/mount_symlink left at their defaults (False), i.e.
        # only VINE_TRANSFER_ALWAYS is set in the flags word.
        FAKE_CVINE.vine_task_add_input.assert_called_once_with(
            t._task, f._file, "remote_in.txt", FAKE_CVINE.VINE_TRANSFER_ALWAYS
        )

    def test_inputs_dict_value_passes_through_extra_parameters(self):
        f = make_file("in2")
        t = Task("echo hello", inputs={f: {"remote_name": "remote_in2.txt", "strict_input": True}})

        self.assertEqual(t._tracked_inputs, [(f, "remote_in2.txt")])

        expected_flags = FAKE_CVINE.VINE_TRANSFER_ALWAYS | FAKE_CVINE.VINE_FIXED_LOCATION
        FAKE_CVINE.vine_task_add_input.assert_called_once_with(
            t._task, f._file, "remote_in2.txt", expected_flags
        )

    def test_inputs_key_not_a_file_raises_typeerror(self):
        with self.assertRaises(TypeError):
            Task("echo hello", inputs={"not_a_file_object": "remote.txt"})

    def test_inputs_value_of_invalid_type_raises_typeerror(self):
        f = make_file("in3")
        with self.assertRaises(TypeError):
            Task("echo hello", inputs={f: 12345})

    def test_outputs_str_value_is_normalized_but_registration_is_deferred(self):
        f = make_file("out1")
        t = Task("echo hello", outputs={f: "remote_out.txt"})

        # add_output() defers the actual cvine registration to
        # _finalize_outputs() (see the comment in task.py above _finalize_outputs);
        # constructing the Task must NOT touch the C layer yet.
        self.assertEqual(t._tracked_outputs, [(f, "remote_out.txt", FAKE_CVINE.VINE_TRANSFER_ALWAYS)])
        self.assertFalse(t._outputs_finalized)
        FAKE_CVINE.vine_task_add_output.assert_not_called()

    def test_outputs_key_not_a_file_raises_typeerror(self):
        with self.assertRaises(TypeError):
            Task("echo hello", outputs={"not_a_file_object": "remote.txt"})

    def test_outputs_value_of_invalid_type_raises_typeerror(self):
        f = make_file("out2")
        with self.assertRaises(TypeError):
            Task("echo hello", outputs={f: 12345})


class TestDeferredOutputRegistration(unittest.TestCase):
    """Task.add_output()/_finalize_outputs(): deferred-registration bookkeeping
    and idempotency (a task's outputs must be registered with the C layer
    exactly once, even if _finalize_outputs() is invoked multiple times)."""

    def setUp(self):
        FAKE_CVINE.vine_task_add_output.reset_mock()

    def test_add_output_only_records_it_does_not_call_cvine(self):
        t = Task("echo hello")
        f = make_file("out3")

        t.add_output(f, "result.txt", watch=True)

        expected_flags = FAKE_CVINE.VINE_TRANSFER_ALWAYS | FAKE_CVINE.VINE_WATCH
        self.assertEqual(t._tracked_outputs, [(f, "result.txt", expected_flags)])
        self.assertFalse(t._outputs_finalized)
        FAKE_CVINE.vine_task_add_output.assert_not_called()

    def test_finalize_outputs_registers_each_tracked_output_once(self):
        t = Task("echo hello")
        f1 = make_file("out4")
        f2 = make_file("out5")
        t.add_output(f1, "one.txt")
        t.add_output(f2, "two.txt", success_only=True)

        t._finalize_outputs()

        self.assertTrue(t._outputs_finalized)
        self.assertEqual(FAKE_CVINE.vine_task_add_output.call_count, 2)

        expected_flags_plain = FAKE_CVINE.VINE_TRANSFER_ALWAYS
        expected_flags_success_only = FAKE_CVINE.VINE_TRANSFER_ALWAYS | FAKE_CVINE.VINE_SUCCESS_ONLY
        FAKE_CVINE.vine_task_add_output.assert_any_call(t._task, f1._file, "one.txt", expected_flags_plain)
        FAKE_CVINE.vine_task_add_output.assert_any_call(t._task, f2._file, "two.txt", expected_flags_success_only)

    def test_finalize_outputs_is_idempotent(self):
        t = Task("echo hello")
        f = make_file("out6")
        t.add_output(f, "result.txt")

        t._finalize_outputs()
        self.assertEqual(FAKE_CVINE.vine_task_add_output.call_count, 1)

        # Calling _finalize_outputs() again (e.g. Manager.submit() being
        # invoked more than once on the same task) must not re-register
        # (and must not double the caller's C-side output list).
        t._finalize_outputs()
        t._finalize_outputs()
        self.assertEqual(FAKE_CVINE.vine_task_add_output.call_count, 1)


class TestStaticFlagHelpers(unittest.TestCase):
    """Task._determine_mount_flags / _determine_file_flags / _determine_cache_level:
    pure functions with real branching logic, independent of any C call."""

    def test_determine_mount_flags_combines_requested_bits(self):
        flags = Task._determine_mount_flags(
            watch=True, failure_only=False, success_only=True, strict_input=True, mount_symlink=False
        )
        expected = (
            FAKE_CVINE.VINE_TRANSFER_ALWAYS
            | FAKE_CVINE.VINE_WATCH
            | FAKE_CVINE.VINE_SUCCESS_ONLY
            | FAKE_CVINE.VINE_FIXED_LOCATION
        )
        self.assertEqual(flags, expected)

    def test_determine_mount_flags_defaults_to_transfer_always_only(self):
        self.assertEqual(Task._determine_mount_flags(), FAKE_CVINE.VINE_TRANSFER_ALWAYS)

    def test_determine_file_flags_peer_noshare_is_default(self):
        # peer_transfer=False (the default) means "do not share with peers",
        # i.e. VINE_PEER_NOSHARE should be set.
        self.assertEqual(Task._determine_file_flags(), FAKE_CVINE.VINE_PEER_NOSHARE)

    def test_determine_file_flags_peer_transfer_clears_noshare_bit(self):
        flags = Task._determine_file_flags(peer_transfer=True, unlink_when_done=True)
        self.assertEqual(flags, FAKE_CVINE.VINE_UNLINK_WHEN_DONE)

    def test_determine_cache_level_true_and_workflow_string(self):
        self.assertEqual(Task._determine_cache_level(True), FAKE_CVINE.VINE_CACHE_LEVEL_WORKFLOW)
        self.assertEqual(Task._determine_cache_level("workflow"), FAKE_CVINE.VINE_CACHE_LEVEL_WORKFLOW)

    def test_determine_cache_level_worker_and_forever(self):
        self.assertEqual(Task._determine_cache_level("worker"), FAKE_CVINE.VINE_CACHE_LEVEL_WORKER)
        self.assertEqual(Task._determine_cache_level("forever"), FAKE_CVINE.VINE_CACHE_LEVEL_FOREVER)

    def test_determine_cache_level_falsy_and_task_string_default_to_task(self):
        self.assertEqual(Task._determine_cache_level(False), FAKE_CVINE.VINE_CACHE_LEVEL_TASK)
        self.assertEqual(Task._determine_cache_level(None), FAKE_CVINE.VINE_CACHE_LEVEL_TASK)
        self.assertEqual(Task._determine_cache_level("task"), FAKE_CVINE.VINE_CACHE_LEVEL_TASK)

    def test_determine_cache_level_invalid_string_raises_valueerror(self):
        with self.assertRaises(ValueError):
            Task._determine_cache_level("not-a-real-level")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# vim: set sts=4 sw=4 ts=4 expandtab ft=python:
