// cloneNetwork: copy any concrete Network by its runtime type.

#include "stdafx.h" // For MSVC, must be first!

#include <typeinfo>

#include "netclone.h"
#include "bareprop.h"
#include "simpleprop.h"
#include "backprop.h"
#include "logistic.h"

using namespace std;

// Makes a copy of the incoming Network (add here if you add a new Network
//    type). The copy constructors take care of everything the type owns --
//    weights, architecture, its DataSet copy -- and Iterative::copy() nulls
//    the observer, so a clone can never drive its original's GUI buffers.
unique_ptr< Network > cloneNetwork( const Network& source )
{
	if ( typeid( source ) == typeid( BareProp ) )
		return make_unique< BareProp >( dynamic_cast< const BareProp& >( source ) );

	if ( typeid( source ) == typeid( SimpleProp ) )
		return make_unique< SimpleProp >( dynamic_cast< const SimpleProp& >( source ) );

	if ( typeid( source ) == typeid( BackProp ) )
		return make_unique< BackProp >( dynamic_cast< const BackProp& >( source ) );

	if ( typeid( source ) == typeid( Logistic ) )
		return make_unique< Logistic >( dynamic_cast< const Logistic& >( source ) );

	return nullptr; // no identifiable Network type was found
}
