// cloneNetwork: copy any concrete Network by its runtime type.
//
// The typeid dispatch lived inside RegressNet::copy_network() from the
// stepwise-regression era; auto algorithm selection needs the same clone
// (three probes from identical weights), so the dispatch is generalized
// here and RegressNet calls it. Add a case when you add a Network type.

#include <memory>

#include "network.h"

#ifndef NETCLONE_H
#define NETCLONE_H

// Returns a deep copy of the passed Network (its own DataSet copy included,
//    via the copy constructors), or nullptr if the concrete type is unknown
std::unique_ptr< Network > cloneNetwork( const Network& source );

#endif
