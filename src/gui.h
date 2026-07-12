// The embedded web GUI: a loopback-only HTTP server (OS-assigned port)
//    serving a single self-contained page that drives the engine

#ifndef GUI_H
#define GUI_H

// Starts the server, prints the URL, optionally opens the browser, and
//    blocks until the process is interrupted. Returns a process exit code.
int run_gui( bool openBrowser );

#endif
