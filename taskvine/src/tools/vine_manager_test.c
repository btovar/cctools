/*
Copyright (C) 2026- The University of Notre Dame
This software is distributed under the GNU General Public License.
See the file COPYING for details.
*/

/*
vine_manager_test is a true unit test (no manager/worker network traffic
involved): it links directly against libtaskvine.a and calls a handful of
vine_manager.c functions with hand-constructed minimal structs, following
the convention set by dttools/src/category_test.c,
dttools/src/histogram_test.c and dttools/src/priority_queue_test.c.

It currently covers:
  - overcommitted_resource_total(): the rounding/threshold arithmetic used
    to compute how much of a worker's resources may be overcommitted.
  - vine_manager_transfer_time(): the bandwidth-estimate + minimum-timeout
    logic used to size manager<->worker transfer timeouts.
  - check_worker_fit(): the fits-in-resources check used to decide whether
    a worker's total resources can satisfy a task's request.

This is a starting point for taskvine unit-test coverage of vine_manager.c
and vine_worker.c -- see the taskvine tech-debt audit for the larger plan.
More scenarios can be added here as coverage grows.
*/

#include "vine_manager.h"
#include "vine_worker_info.h"
#include "vine_resources.h"

#include "rmsummary.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(desc, cond)                                                                                                                                                                            \
	do {                                                                                                                                                                                            \
		if (cond) {                                                                                                                                                                                \
			printf("PASS: %s\n", desc);                                                                                                                                                           \
		} else {                                                                                                                                                                                   \
			printf("FAIL: %s\n", desc);                                                                                                                                                           \
			failures++;                                                                                                                                                                            \
		}                                                                                                                                                                                          \
	} while (0)

/*
overcommitted_resource_total() rounds total*resource_submit_multiplier up
to the nearest integer, except that a total of exactly 0 always yields 0
regardless of the multiplier.
*/
static void test_overcommitted_resource_total()
{
	struct vine_manager q;
	memset(&q, 0, sizeof(q));

	/* A total of 0 is a special case: always returns 0, whatever the multiplier is. */
	q.resource_submit_multiplier = 1.75;
	CHECK("overcommitted_resource_total: total=0 returns 0 regardless of multiplier", overcommitted_resource_total(&q, 0) == 0);

	/* multiplier of 1.0 should not change the total. */
	q.resource_submit_multiplier = 1.0;
	CHECK("overcommitted_resource_total: multiplier=1.0 is a no-op", overcommitted_resource_total(&q, 7) == 7);

	/* Fractional results must round UP (ceil), not truncate. */
	q.resource_submit_multiplier = 1.5;
	CHECK("overcommitted_resource_total: 10 * 1.5 == 15 exactly", overcommitted_resource_total(&q, 10) == 15);
	CHECK("overcommitted_resource_total: 1 * 1.5 rounds up to 2, not 1", overcommitted_resource_total(&q, 1) == 2);
	CHECK("overcommitted_resource_total: 3 * 1.5 == 4.5 rounds up to 5", overcommitted_resource_total(&q, 3) == 5);
}

/*
vine_manager_transfer_time() picks a bandwidth estimate (worker's observed
rate if available, otherwise the manager's own observed rate, otherwise a
configured default), divides it by the transfer_outlier_factor to get a
"tolerable" rate, computes length/tolerable_rate, and finally clamps the
result to be no less than minimum_transfer_timeout.
*/
static void test_vine_manager_transfer_time()
{
	struct vine_manager q;
	struct vine_stats stats;
	struct vine_worker_info w;

	memset(&q, 0, sizeof(q));
	memset(&stats, 0, sizeof(stats));
	memset(&w, 0, sizeof(w));

	q.stats = &stats;
	q.minimum_transfer_timeout = 3;
	q.transfer_outlier_factor = 10;
	q.default_transfer_rate = 1000000; /* 1 MB/s, used only when no observations are available */

	w.hostname = "test-worker";
	w.addrport = "127.0.0.1:9123";

	/* Case 1: no observed history anywhere, so the manager's configured default rate is used.
	 * tolerable = 1000000/10 = 100000 B/s; timeout = 5000000/100000 = 50s; that's above the minimum. */
	w.total_transfer_time = 0;
	w.total_bytes_transferred = 0;
	stats.time_send = 0;
	stats.time_receive = 0;
	CHECK("vine_manager_transfer_time: uses default rate absent any observations", vine_manager_transfer_time(&q, &w, 5000000) == 50);

	/* Case 2: worker has enough observed history (>1s of transfer time) to use its own rate.
	 * 4,000,000 bytes / 2,000,000 us = 2,000,000 B/s observed; tolerable = 200000 B/s;
	 * timeout = 2,000,000/200000 = 10s. */
	w.total_transfer_time = 2000000; /* microseconds */
	w.total_bytes_transferred = 4000000;
	CHECK("vine_manager_transfer_time: prefers the worker's own observed rate", vine_manager_transfer_time(&q, &w, 2000000) == 10);

	/* Case 3: worker has no history, but the manager as a whole does (>1s of aggregate transfer time).
	 * 8,000,000 bytes / 4,000,000 us = 2,000,000 B/s; tolerable = 200000 B/s; timeout = 1,000,000/200000 = 5s. */
	w.total_transfer_time = 0;
	w.total_bytes_transferred = 0;
	stats.bytes_sent = 5000000;
	stats.bytes_received = 3000000;
	stats.time_send = 3000000;
	stats.time_receive = 1000000;
	CHECK("vine_manager_transfer_time: falls back to the manager's aggregate observed rate", vine_manager_transfer_time(&q, &w, 1000000) == 5);

	/* Case 4: an extremely fast rate would compute a sub-minimum timeout; the minimum must win. */
	stats.bytes_sent = 0;
	stats.bytes_received = 0;
	stats.time_send = 0;
	stats.time_receive = 0;
	q.default_transfer_rate = 1000000000; /* 1 GB/s */
	CHECK("vine_manager_transfer_time: never returns less than minimum_transfer_timeout", vine_manager_transfer_time(&q, &w, 1) == q.minimum_transfer_timeout);
}

/*
check_worker_fit() returns the worker's total worker-slot count if the
given resource request fits within the worker's *total* resources (not
what is currently in use), 0 otherwise. A NULL request only checks that
the worker has at least one worker slot.
*/
static void test_check_worker_fit()
{
	struct vine_worker_info w;
	memset(&w, 0, sizeof(w));
	w.resources = vine_resources_create();

	/* A worker advertising zero worker slots can never fit anything, even an empty request. */
	w.resources->workers.total = 0;
	w.resources->cores.total = 4;
	w.resources->memory.total = 8000;
	w.resources->disk.total = 10000;
	w.resources->gpus.total = 0;
	CHECK("check_worker_fit: a worker with 0 worker slots never fits, even a NULL request", check_worker_fit(&w, NULL) == 0);

	w.resources->workers.total = 1;

	/* A NULL request just asks "does this worker have any slots at all" -- returns the slot count. */
	CHECK("check_worker_fit: NULL request returns the worker's slot count", check_worker_fit(&w, NULL) == 1);

	struct rmsummary *s = rmsummary_create(-1);

	/* A request that fits exactly at the boundary (== total, not >) must be accepted. */
	s->cores = 4;
	s->memory = 8000;
	s->disk = 10000;
	s->gpus = 0;
	CHECK("check_worker_fit: a request exactly equal to worker totals fits", check_worker_fit(&w, s) == 1);

	/* Exceeding cores alone must reject the request. */
	s->cores = 5;
	CHECK("check_worker_fit: exceeding cores alone rejects the request", check_worker_fit(&w, s) == 0);
	s->cores = 4;

	/* Exceeding memory alone must reject the request. */
	s->memory = 8001;
	CHECK("check_worker_fit: exceeding memory alone rejects the request", check_worker_fit(&w, s) == 0);
	s->memory = 8000;

	/* Exceeding gpus alone must reject the request, even when cores/memory/disk all fit. */
	s->gpus = 1;
	CHECK("check_worker_fit: exceeding gpus alone rejects the request", check_worker_fit(&w, s) == 0);
	s->gpus = 0;

	/* Sanity check: a comfortably smaller request still fits. */
	s->cores = 1;
	s->memory = 100;
	s->disk = 100;
	s->gpus = 0;
	CHECK("check_worker_fit: a comfortably smaller request fits", check_worker_fit(&w, s) == 1);

	rmsummary_delete(s);
	vine_resources_delete(w.resources);
}

int main(int argc, char *argv[])
{
	test_overcommitted_resource_total();
	test_vine_manager_transfer_time();
	test_check_worker_fit();

	if (failures) {
		fprintf(stderr, "vine_manager_test: %d check(s) failed\n", failures);
		return 1;
	}

	printf("vine_manager_test: all checks passed\n");
	return 0;
}

/* vim: set noexpandtab tabstop=8: */
