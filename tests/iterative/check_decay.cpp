// check_decay.cpp : weight decay is a property of lambda and eta, not of N.
//
// THE DEFECT (D4, refactor_audit.md section 9). The canonical decay multiplier
//
//     w *= decayTerm;          // decayTerm = 1 - eta*decay = 1 - 2*eta*lambda
//
// sat ABOVE the batch/on-line split in every model's innerTrainSet, so it ran
// once per EXEMPLAR in both modes. On-line makes one weight update per exemplar,
// so once per exemplar is right there. Batch makes ONE update per epoch, so the
// implemented per-epoch factor was
//
//     ( 1 - eta*decay )^N     instead of     ( 1 - eta*decay )
//
// -- exponential in the number of training rows. The manifest's own formula,
// quoted in the source beside it, is
//
//     w_{t+1} = ( 1 - 2*eta*lambda ) w_t - eta * dE/dw
//
// with the factor applied once per UPDATE.
//
// THE TEST, and why duplication rather than two different datasets. Duplicating
// every observation leaves the MEAN gradient identical -- each row contributes
// twice and the accumulator is divided by 2N -- so a correct batch epoch must
// reach the same weights from N rows or from 2N duplicated rows. Under the
// defect it cannot: the decay exponent doubles. Comparing two DIFFERENT-sized
// datasets would confound the decay rule with the data, which is the mistake an
// earlier draft of this measurement made.
//
// Both models are evaluated on the SAME input matrix, so the comparison is of
// weights, not of whatever rows each happened to hold.
//
// COVERAGE: SimpleProp, BareProp, BackProp and Logistic -- the defect is
// structurally identical in all four, so the fix must be too.
//
// SABOTAGE: move the multiplier back above the batch/on-line split and the
// decay-ON invariance fails in all four; the decay-OFF control still passes,
// which is exactly why this went unnoticed.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "simpleprop.h"
#include "bareprop.h"
#include "backprop.h"
#include "logistic.h"
#include "dataset.h"
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

struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

// Each probe exposes forward propagation over an ARBITRARY input matrix, so two
// models trained on differently sized sets can be compared on common inputs.
#define PROBE( NAME, BASE ) \
	class NAME : public BASE { \
	public: \
		double at( Matrix< double >& m, unsigned r ) { forward( m, r ); return o; } \
		Matrix< double >& inputs() { return Train; } \
	};

PROBE( ProbeSimple, SimpleProp )
PROBE( ProbeBare, BareProp )
PROBE( ProbeBack, BackProp )
PROBE( ProbeLogistic, Logistic )

// raw2train, not randomize: no shuffle and no test set, so the duplicated set's
// rows stay in a known relation to the original's. Normalization is unaffected
// by duplication (min and max are the same), so both sets scale identically.
static DataSet makeData( unsigned n, bool duplicated )
{
	unsigned rows = duplicated ? n * 2 : n;
	Matrix< double > raw( rows, 3 );
	for ( unsigned i = 0; i < rows; i++ )
	{
		unsigned k = duplicated ? ( i / 2 ) : i; // each row twice when duplicated
		double x0 = -1.0 + 2.0 * ( ( k * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( k * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0.55 ) ? 1 : 0;
	}

	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	Hush quiet;
	d.setRawMatrix( raw );
	d.raw2train();
	return d;
}

// Configure identically; the ONLY difference between the two runs of a pair is
// how many (duplicated) rows the training set holds.
template < class NET >
static void configure( NET& net, bool decay, bool batch )
{
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setQuiet( true );
	net.setXEerror();
	net.setWeightDecay( decay );
	net.setDecay( decay ? 0.01 : 0 ); // a lambda large enough to be visible
	net.setBatchEpoch( batch );
	net.setAutoStepSize( false );
	net.setEta( 1.0 );
	net.setGradStop( false ); // the branch that uses the multiplier
	net.setMaxIterations( 100 );
}

// Train on n rows and on 2n duplicated rows; compare the two models' outputs
// over the SAME inputs. relTol allows for the reordered floating-point sum in
// the larger accumulator -- the mean gradient is mathematically identical, but
// it is not summed in the same order.
template < class PROBE >
static double duplicationGap( unsigned n, bool decay, bool batch,
	unsigned hidden )
{
	util::set_seed( 4242 );
	DataSet dOne = makeData( n, false );
	util::set_seed( 4242 );
	DataSet dTwo = makeData( n, true );

	PROBE one, two;
	one.setDataSet( dOne );
	two.setDataSet( dTwo );
	one.setHidden( hidden );
	two.setHidden( hidden );
	configure( one, decay, batch );
	configure( two, decay, batch );

	util::set_seed( 7 ); one.randomize();
	util::set_seed( 7 ); two.randomize(); // identical initial weights

	{ Hush h; one.train(); }
	{ Hush h; two.train(); }

	// Both evaluated on the SMALLER set's inputs: a comparison of weights
	Matrix< double >& common = one.inputs();
	double worst = 0;
	for ( unsigned r = 0; r < common.rows(); r++ )
	{
		double a = one.at( common, r ), b = two.at( common, r );
		double gap = fabs( a - b );
		if ( gap > worst ) worst = gap;
	}
	return worst;
}

// Logistic takes no hidden layer, and is batch by definition
static double duplicationGapLogistic( unsigned n, bool decay )
{
	util::set_seed( 4242 );
	DataSet dOne = makeData( n, false );
	util::set_seed( 4242 );
	DataSet dTwo = makeData( n, true );

	ProbeLogistic one, two;
	one.setDataSet( dOne );
	two.setDataSet( dTwo );
	configure( one, decay, true );
	configure( two, decay, true );

	util::set_seed( 7 ); one.randomize();
	util::set_seed( 7 ); two.randomize();

	{ Hush h; one.train(); }
	{ Hush h; two.train(); }

	Matrix< double >& common = one.inputs();
	double worst = 0;
	for ( unsigned r = 0; r < common.rows(); r++ )
	{
		double gap = fabs( one.at( common, r ) - two.at( common, r ) );
		if ( gap > worst ) worst = gap;
	}
	return worst;
}

// BackProp needs its architecture as a vector, and its bias set before the data
static double duplicationGapBackProp( unsigned n, bool decay )
{
	util::set_seed( 4242 );
	DataSet dOne = makeData( n, false );
	util::set_seed( 4242 );
	DataSet dTwo = makeData( n, true );

	ProbeBack one, two;
	one.setBias( true ); two.setBias( true ); // BEFORE setDataSet
	one.setDataSet( dOne );
	two.setDataSet( dTwo );
	vector< unsigned > layers;
	layers.push_back( 3 );
	layers.push_back( 2 ); // two hidden layers: a genuine BackProp
	one.setHidden( layers );
	two.setHidden( layers );
	configure( one, decay, true );
	configure( two, decay, true );

	util::set_seed( 7 ); one.randomize();
	util::set_seed( 7 ); two.randomize();

	{ Hush h; one.train(); }
	{ Hush h; two.train(); }

	Matrix< double >& common = one.inputs();
	double worst = 0;
	for ( unsigned r = 0; r < common.rows(); r++ )
	{
		double gap = fabs( one.at( common, r ) - two.at( common, r ) );
		if ( gap > worst ) worst = gap;
	}
	return worst;
}

// The mean gradient is identical under duplication; only the summation ORDER
// differs, so agreement should be at rounding level. 1e-9 is far below any
// decay-rule effect (which is a factor of decayTerm^N) and far above the
// accumulated rounding of a few hundred additions.
static const double TOL = 1e-9;

static void check( double gap, const string& what )
{
	if ( gap >= TOL )
		cout << "         worst output gap " << setprecision( 6 ) << gap
			<< " (tolerance " << TOL << ")" << endl;
	expect( gap < TOL, what );
}

int main()
{
	cout << "-- batch epoch: duplicating every row must not change the fit --" << endl;

	check( duplicationGap< ProbeSimple >( 60, true, true, 3 ),
		"SimpleProp, weight decay ON" );
	check( duplicationGap< ProbeBare >( 60, true, true, 3 ),
		"BareProp, weight decay ON" );
	check( duplicationGapBackProp( 60, true ),
		"BackProp, weight decay ON" );
	check( duplicationGapLogistic( 60, true ),
		"Logistic, weight decay ON" );

	cout << "-- the control: with decay off, the invariance already held --" << endl;

	check( duplicationGap< ProbeSimple >( 60, false, true, 3 ),
		"SimpleProp, weight decay OFF" );
	check( duplicationGapLogistic( 60, false ),
		"Logistic, weight decay OFF" );

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
