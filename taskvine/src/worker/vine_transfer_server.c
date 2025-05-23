/*
Copyright (C) 2022- The University of Notre Dame
This software is distributed under the GNU General Public License.
See the file COPYING for details.
*/

#include "vine_transfer_server.h"
#include "vine_protocol.h"
#include "vine_transfer.h"
#include "vine_worker.h"

#include "change_process_title.h"
#include "debug.h"
#include "link.h"
#include "link_auth.h"
#include "process.h"
#include "url_encode.h"

#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* The initial timeout to wait for a command is short, to avoid unnecessary hangs */
static int command_timeout = 5;

/* The timeout to handle a valid transfer is much higher, to avoid false failures. */
static int transfer_timeout = 3600;

/* The server link from which connections are accepted */
static struct link *transfer_link = 0;

/* Pid of process handling peer transfers. */
pid_t transfer_server_pid = 0;

/* Specific port for the transfer server to listen on.  Zero means choose any available. */
int vine_transfer_server_port = 0;

/* Handle a single request for a transfer request from a peer. */

static void vine_transfer_handler(struct link *lnk, struct vine_cache *cache)
{
	char line[VINE_LINE_MAX];
	char filename_encoded[VINE_LINE_MAX];
	char filename[VINE_LINE_MAX];

	change_process_title("vine_worker [transfer]");

	if (options->password) {
		if (!link_auth_password(lnk, options->password, time(0) + command_timeout)) {
			debug(D_VINE, "transfer server: could not authenticate peer worker via password!");
			return;
		}
	}

	if (link_readline(lnk, line, sizeof(line), time(0) + command_timeout)) {
		if (sscanf(line, "get %s", filename_encoded) == 1) {
			url_decode(filename_encoded, filename, sizeof(filename));
			vine_transfer_put_any(lnk, cache, filename, VINE_TRANSFER_MODE_ANY, time(0) + transfer_timeout);
		} else {
			debug(D_VINE, "invalid peer transfer message: %s\n", line);
		}
	}
}

static void vine_transfer_process(struct vine_cache *cache)
{
	static int child_count = 0;

	/*
	1. Perform a non-blocking check for any child processes that have exited, this runs very fast.
	2. If the number of child processes has reached the maximum allowed, perform a blocking wait for a child process to exit.
	3. Once arrives here, the server is safe to accept a new connection, as the child_count is less than the maximum allowed.
	4. Upon accepting a connection, fork a new child process to handle it; if timeout, simply continue.
	5. lnk should be closed in the parent process to prevent file descriptor exhaustion.
	*/
	while (1) {
		/* Do a non-blocking wait for any exited children. */
		while (waitpid(-1, NULL, WNOHANG) > 0) {
			child_count--;
		}
		/* If the child count is larger than the maximum allowed, do blocking wait until it is safe to accept a new connection. */
		while (child_count >= VINE_TRANSFER_PROC_MAX_CHILD) {
			debug(D_VINE, "Transfer Server: waiting on exited child. Reached %d", child_count);
			if (waitpid(-1, NULL, 0) > 0) {
				child_count--;
			}
		}

		/* The server is safe to accept a new connection. */
		struct link *lnk = link_accept(transfer_link, time(0) + 10);

		if (lnk) {
			pid_t p = fork();
			if (p == 0) {
				vine_transfer_handler(lnk, cache);
				link_close(lnk);
				_exit(0);
			} else if (p > 0) {
				/* Increment the child count when a new child is successfully forked. */
				child_count++;
				/* Also close the link in the parent process, otherwise the opened file descriptors will not be closed.
				 * This caused a problem where incoming transfers were all failing due to the file descriptor limit per process being reached. */
				link_close(lnk);
			} else {
				/* If fork fails, also close the link. */
				link_close(lnk);
			}
		} else {
			/* If lnk is NULL, it means link_accept failed to accept a connection.
			 * This could be due to a timeout or other transient issues. */
			continue;
		}
	}
}

void vine_transfer_server_start(struct vine_cache *cache, int port_min, int port_max)
{
	transfer_link = link_serve_range(port_min, port_max);

	if (!transfer_link) {
		fatal("unable to find a port to start a transfer server.");
	}

	transfer_server_pid = fork();
	if (transfer_server_pid == 0) {
		// consider closing additional resources here?
		change_process_title("vine_worker [transfer server]");
		vine_transfer_process(cache);
		_exit(0);
	} else if (transfer_server_pid > 0) {
		char addr[LINK_ADDRESS_MAX];
		int port;
		vine_transfer_server_address(addr, &port);
		debug(D_VINE, "started transfer server pid %d listening on %s:%d", transfer_server_pid, addr, port);
		// in parent, keep going
	} else {
		fatal("unable to fork transfer server: %s", strerror(errno));
	}
}

void vine_transfer_server_stop()
{
	int status;

	debug(D_VINE, "stopping transfer server pid %d", transfer_server_pid);

	link_close(transfer_link);
	kill(transfer_server_pid, SIGKILL);
	waitpid(transfer_server_pid, &status, 0);

	transfer_server_pid = 0;
	transfer_link = 0;
}

void vine_transfer_server_address(char *addr, int *port)
{
	link_address_local(transfer_link, addr, port);
}
