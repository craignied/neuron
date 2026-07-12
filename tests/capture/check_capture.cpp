// Verifies util::set_screen(): engine-core output must be capturable into a
// string (the GUI/server contract) and cout must be restorable. Exercises
// both output paths: prints routed internally through util::screen()
// (DataSet's refusal message) and report methods taking an ostream&
// parameter (TwoSet::ClassTable). No GSL needed.

#include <iostream>
#include <sstream>
#include <string>

#include "dataset.h"
#include "twoset.h"
#include "utility.h"

using namespace std;

int failures = 0;

void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

int main()
{
	// --- Path 1: engine prints routed through util::screen() -------------
	ostringstream captured;
	util::set_screen( captured );

	// A 5-column Matrix into a DataSet expecting 1 input + 1 output makes
	// the engine print its refusal — through util::screen()
	Matrix< double > wrong( 4, 5 );
	DataSet d;
	d.setRawMatrix( wrong );

	util::set_screen( cout ); // restore before reporting results

	expect( captured.str().find( "I can't load the Matrix" ) != string::npos,
		"engine refusal message was captured, not printed" );

	// --- Path 2: report methods taking an explicit ostream& --------------
	Matrix< double > A( 8, 2 );
	for ( unsigned r = 0; r < 8; r++ )
	{
		A( r, 0 ) = ( r < 4 ? 0 : 1 ); // outcome
		A( r, 1 ) = ( r < 4 ? 0.1 + 0.01 * r : 0.8 + 0.01 * r ); // guess
	}
	TwoSet t;
	t.setMatrix( A );
	t.setThreshold( 0.5 );

	ostringstream table;
	t.ClassTable( table );
	expect( table.str().find( "Predicted" ) != string::npos,
		"ClassTable report captured via ostream& parameter" );

	// --- ROC curve-point capture (Phase 3.3 port from the roc app) -------
	double area = t.getTrapROCarea();
	const vector< double >& rx = t.getROCx();
	const vector< double >& ry = t.getROCy();

	expect( area > 0.99, "cleanly separated data gives ROC area ~1" );
	expect( !rx.empty() && rx.size() == ry.size(),
		"getTrapROCarea captured matching ROC curve point vectors" );

	bool inRange = true;
	for ( unsigned i = 0; i < rx.size(); i++ )
		if ( rx[ i ] < 0 || rx[ i ] > 1 || ry[ i ] < 0 || ry[ i ] > 1 )
			inRange = false;
	expect( inRange, "all captured curve points lie in the unit square" );

	// --- Restoration ------------------------------------------------------
	expect( &util::screen() == static_cast< ostream* >( &cout ),
		"set_screen( cout ) restores the default stream" );

	return failures == 0 ? 0 : 1;
}
