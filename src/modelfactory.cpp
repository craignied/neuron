/* Model construction service -- see modelfactory.h and rule 6. */

#include "modelfactory.h"

#include "network.h"
#include "logistic.h"
#include "simpleprop.h"
#include "bareprop.h"
#include "backprop.h"

unique_ptr< Model > modelfactory::build( const Spec& spec, DataSet& data,
	string& err )
{
	unique_ptr< Model > m;

	if ( spec.logistic )
	{
		// The CLI's own check: exactly 1 output, and it must be discrete.
		if ( data.getOutput() != 1 || !data.getDiscrete() )
		{
			err = "for logistic regression, there can be only 1 output, "
				"and it must be discrete";
			return nullptr;
		}
		m = make_unique< Logistic >();
		m->setDataSet( data );
		m->setLastop( spec.logLastop );
		m->setHistory( spec.logHistory );
		return m;
	}

	if ( spec.hidden.empty() )
	{
		err = "a neural network needs at least one hidden layer";
		return nullptr;
	}

	// Neural network: 1 output AND 1 hidden layer -> SimpleProp (with bias) or
	//    BareProp (without); anything else -> a general BackProp. The BackProp
	//    bias must be set BEFORE setDataSet; the architecture (setHidden) AFTER
	//    it -- the manifest construction order.
	bool single = ( data.getOutput() == 1 && spec.hidden.size() == 1 );

	if ( single && spec.bias ) m = make_unique< SimpleProp >();
	else if ( single )         m = make_unique< BareProp >();
	else
	{
		m = make_unique< BackProp >();
		dynamic_cast< Network* >( m.get() )->setBias( spec.bias );
	}

	m->setDataSet( data ); // BEFORE the architecture
	spec.xentropy ? m->setXEerror() : m->setLMSerror();

	if ( single && spec.bias )
		dynamic_cast< SimpleProp* >( m.get() )->setHidden( spec.hidden[ 0 ] );
	else if ( single )
		dynamic_cast< BareProp* >( m.get() )->setHidden( spec.hidden[ 0 ] );
	else
		dynamic_cast< BackProp* >( m.get() )->setHidden( spec.hidden );

	m->setLastop( spec.logLastop );
	m->setHistory( spec.logHistory );

	return m;
}

unique_ptr< Model > modelfactory::createByTypeName( const string& typeName,
	bool bias )
{
	unique_ptr< Model > m;

	if ( typeName == "Binary logistic" ) m = make_unique< Logistic >();
	else if ( typeName == "SimpleProp" ) m = make_unique< SimpleProp >();
	else if ( typeName == "BareProp" )   m = make_unique< BareProp >();
	else if ( typeName == "BackProp" )
	{
		m = make_unique< BackProp >();
		dynamic_cast< Network* >( m.get() )->setBias( bias );
	}

	return m; // nullptr for an unrecognized type
}
