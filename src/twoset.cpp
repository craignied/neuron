// Member functions of TwoSet class, which represents a dataset with
// one variable and one outcome.

#include "stdafx.h" // For MSVC++ MUST BE FIRST!

#include "twoset.h"

// Default constructor
TwoSet::TwoSet() : n ( 0 ), loadedFlag ( false )
{
	initialize();
}

// Constructor loads Matrix
TwoSet::TwoSet( const Matrix< double >& D )
{
	initialize();
	setMatrix( D );
}

// Initial values for constructors
void TwoSet::initialize()
{
	thresholdFlag = false;
	nBins = 10;
	binSize = 10;
	binThresh = 100;
	nBinFlag = true;
	binnedFlag = false;
	statROCcalcFlag = false;
	searchCalcFlag = false;
	searchErrorFlag = false;
	searchErrorMsg.clear();
	bestPfit = ROCfit();
	bestAUCfit = ROCfit();
	bootCalcFlag = false;
	bootB = 2000; // resamples for the ROC confidence intervals
	bestPci = CI();
	bestAUCci = CI();
	statCi = CI();
	calcThresh = 10;
	reportFlag = true;
	searchFlag = true;
	KScalcFlag = false;
	PKX2calcFlag = false;  // Hui Liu added 09/09/2004
	HLX2calcFlag = false;  //  Hui Liu added 09/09/2004
	minBins = 3;
	maxBins = 10;
	// Zero the cached statistic values so no path can ever print
	//    uninitialized memory (2.x left these uninitialized)
	statP = statChi2 = KSD = KSP = PKX2P = HLX2P = 0;
	statPoints = 0;
}

// Copy constructor
TwoSet::TwoSet( const TwoSet& rhs )
{
	copy( rhs ); // use the copy utility
}

// Overloaded = operator
TwoSet& TwoSet::operator= ( const TwoSet& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		copy( rhs ); // use the copy utility

	return *this; // enables A = B = C
}

// Copy utility
void TwoSet::copy( const TwoSet& rhs )
{
	A = rhs.A;
	n = rhs.n;
	tp = rhs.tp;
	tn = rhs.tn;
	fp = rhs.fp;
	fn = rhs.fn;
	threshold = rhs.threshold;
	nBins = rhs.nBins;
	binSize = rhs.binSize;
	binThresh = rhs.binThresh;
	calcThresh = rhs.calcThresh;
	nThresholds = rhs.nThresholds;
	minBins = rhs.minBins;
	maxBins = rhs.maxBins;
	bootB = rhs.bootB;
	statP = rhs.statP;
	statChi2 = rhs.statChi2;
	statPoints = rhs.statPoints;
	loadedFlag = rhs.loadedFlag;
	thresholdFlag = rhs.thresholdFlag;
	nBinFlag = rhs.nBinFlag;
	binnedFlag = rhs.binnedFlag;
	statROCcalcFlag = rhs.statROCcalcFlag;
	searchCalcFlag = rhs.searchCalcFlag;
	searchErrorFlag = rhs.searchErrorFlag;
	bootCalcFlag = rhs.bootCalcFlag;
	reportFlag = rhs.reportFlag;
	searchFlag = rhs.searchFlag;
	KScalcFlag = rhs.KScalcFlag;
	PKX2calcFlag = rhs.PKX2calcFlag;  // Hui Liu added 09/09/2004
	HLX2calcFlag = rhs.HLX2calcFlag;   // Hui Liu added 09/09/2004
	// Copy the cached statistic values with their flags — 2.x copied the
	//    flags but not the values, so copies claimed statistics they never held
	KSD = rhs.KSD;
	KSP = rhs.KSP;
	PKX2P = rhs.PKX2P;
	HLX2P = rhs.HLX2P;
	bestPfit = rhs.bestPfit;
	bestAUCfit = rhs.bestAUCfit;
	bestPci = rhs.bestPci;
	bestAUCci = rhs.bestAUCci;
	statCi = rhs.statCi;
	searchErrorMsg = rhs.searchErrorMsg;
	ROCx = rhs.ROCx;
	ROCy = rhs.ROCy;
}

// Loads Matrix into TwoSet object, takes incoming Matrix as argument
void TwoSet::setMatrix( const Matrix< double >& inMatrix )
{
	// Bomb out if either dimension of incoming Matrix is zero
	assert ( inMatrix.rows() != 0 && inMatrix.cols() != 0 );

	loadedFlag = false; // initial state of Matrix loaded flag

	if ( inMatrix.cols() != 2 ) // check incoming Matrix columns
		// Inform user that Matrix will not be loaded
		util::screen() << "I'm sorry, but I can't load a non 2-column TwoSet Matrix." << endl;

	else if ( !checkDiscrete( inMatrix ) ) // check to see if real column discrete
		// Inform user that Matrix will not be loaded
		util::screen() << "I'm sorry, but the first column of the Matrix must be discrete."
			<< endl;

	else // Matrix checks out, so complete operation
	{
		A = inMatrix; // set data Matrix to incoming Matrix
		loadedFlag = true; // and the Matrix loaded flag
		n = A.rows(); // and set the number of outcomes
		nThresholds = n; // default number of thresholds for ROC area
		thresholdFlag = false; // reset the threshold
		statROCcalcFlag = false; // ROC has not yet been calculated for new dataset
		KScalcFlag = false; // KS test has not yet been calculated for new dataset
		PKX2calcFlag = false;  // Hui Liu added 09/23/2004
		HLX2calcFlag = false;   // Hui Liu added 09/23/2004
	}
}

// Loads TwoSet Matrix from file, takes filename (string) as argument
void TwoSet::load( string& filename ) // loads from file
{
	// Load the Matrix
	A.loadfile( true, filename );

	this->setMatrix( A ); // use previously coded setMatrix method
}

// Saves TwoSet Matrix to file, takes filename (string) as argument
bool TwoSet::save( string& filename ) // saves to file
{
	bool success = false; // flag to indicate file successfully saved

	if ( !loadedFlag ) // test to see if a TwoSet object has been loaded
		util::screen() << "A TwoSet object must be loaded before it can be saved!" << endl;
	else
	{
		// Open the output file to save data, overwrite file if it exists
		ofstream savefile( filename.c_str(), ios::out | ios::trunc );

		// Test to insure it was opened
		if ( !savefile.is_open() )
			util::screen() << "Error in opening file to save TwoSet object!" << endl;

		else
		{
			// Output the Matrix to the file without a header
			savefile << A.setHeader( false );

			// Print message to user notifying successful save to file
			util::screen() << "The TwoSet object was successfully saved to the file " << filename
				<< "." << endl;

			savefile.close(); // close output file

			success = true; // set flag to indicate file successfully saved
		}
	}
	return success; // return flag to indicate if file successfully saved
}

// Accessor for the "real" column, takes vector to insert in the real column
//    as argument, returns a reference to the TwoSet object
TwoSet& TwoSet::insertReal( const vector< double >& v )
{
	// Matrix will check dimensions, but for debugging, it's helpful here too
	assert( n == v.size() );

	A.replacecol( 0, v ); // replace column 0 (real) with the vector

	return *this; // enables use in TwoSet formulae
}

// Accessor for the "real" column, takes a vector to be populated with that
//    column as an argument, returns a reference to that vector
vector< double >& TwoSet::real( vector< double >& v ) const
{
	// Matrix will check dimensions, but for debugging, it's helpful here too
	assert( n == v.size() );

	A.col( 0, v ); // put column 0 (real) into the vector

	return v; // enables use in vector formulae
}

// Accessor for the "real" column, returns a new vector
vector< double > TwoSet::real() const
{
	// Construct receiving vector
	vector< double > v( n );

	// Use previously coded real method
	this->real( v );

	return v; // enables use in vector formulae
}

// Accessor for the "test" column, takes vector to insert in the test column
//    as argument, returns a reference to the TwoSet object
TwoSet& TwoSet::insertTest( const vector< double >& v )
{
	// Matrix will check dimensions, but for debugging, it's helpful here too
	assert( n == v.size() );

	A.replacecol( 1, v ); // replace column 1 (test) with the vector

	return *this; // enables use in TwoSet formulae
}

// Accessor for the "test" column, takes a vector to be populated with that
//    column as an argument, returns a reference to that vector
vector< double >& TwoSet::test( vector< double >& v ) const
{
	// Matrix will check dimensions, but for debugging, it's helpful here too
	assert( n == v.size() );

	A.col( 1, v ); // put column 1 (test) into the vector

	return v; // enables use in vector formulae
}

// Accessor for the "test" column, returns a new vector
vector< double > TwoSet::test() const
{
	// Construct receiving vector
	vector< double > v( n );

	// Use previously coded test method
	this->test( v );

	return v; // enables use in vector formulae
}

// Accessor for a row's element in the real column
double& TwoSet::real( const unsigned r )
{
	assert( r >= 0 && r < n ); // bounds check row argument

	return A( r, 0 );
}

// Const accessor for a row's element in the real column
const double TwoSet::real( const unsigned r ) const
{
	assert( r >= 0 && r < n ); // bounds check row argument

	return A( r, 0 );
}

// Accessor for a row's element in the test column
double& TwoSet::test( const unsigned r )
{
	assert( r >= 0 && r < n ); // bounds check row argument

	// Handing out a writable reference means the guess column may change,
	//    so cached statistics can no longer be trusted. In 2.x they were
	//    never invalidated: retraining printed the FIRST run's K-S/Pearson/
	//    H-L values, and stepwise regression printed uninitialized memory.
	invalidate();

	return A( r, 1 );
}

// Invalidate cached statistics (call whenever the guess column changes)
void TwoSet::invalidate()
{
	statROCcalcFlag = false;
	searchCalcFlag = false; // the cached binning search describes the old guesses
	bootCalcFlag = false; // and so do the cached bootstrap intervals
	KScalcFlag = false;
	PKX2calcFlag = false;
	HLX2calcFlag = false;
}

// Const accessor for a row's element in the test column
const double TwoSet::test( const unsigned r ) const
{
	assert( r >= 0 && r < n ); // bounds check row argument

	return A( r, 1 );
}

// Accessor (set) for threshold data member of TwoSet ADT
//    takes (double) threshold value as argument
// Calculates true positives, true negatives, false positives
//    and false negatives when called
void TwoSet::setThreshold( const double thresh )
{
	if ( !loadedFlag ) // a Matrix must have been loaded
		util::screen() << "I'm sorry, but a TwoSet Matrix has not yet been defined." << endl;

	else
	{
		threshold = thresh; // set the threshold
		thresholdFlag = true; // a threshold has now been set
	}
}

// Calculates and returns the classification accuracy
double TwoSet::getClassAcc()
{
	double result = 0;

	if ( !thresholdFlag ) // a threshold must have been set
		util::screen() << "Classification accuracy cannot be calculated: a threshold has not been set." << endl;
	else // calculate classification accuracy
	{
		calculate( threshold ); // calculate tp, fp, tn, fn
		result = static_cast< double >( tp + tn ) / n;
	}

	return result;
}

// Calculates and returns the sensitivity
double TwoSet::getSens()
{
	double result = 0;

	if ( !thresholdFlag ) // a threshold must have been set
		util::screen() << "Sensitivity cannot be calculated: a threshold has not been set." << endl;
	else // calculate sensitivity
	{
		calculate( threshold ); // calculate tp, fp, tn, fn
		result = static_cast< double >( tp ) / ( tp + fn );
		if ( ( tp + fn ) == 0 ) // check division by zero
			throw DivisionByZero();
	}

	return result;
}

// Calculates and returns the specificity
double TwoSet::getSpec()
{
	double result = 0;

	if ( !thresholdFlag ) // a threshold must have been set
		util::screen() << "Specificity cannot be calculated: a threshold has not been set." << endl;
	else // calculate specificity
	{
		calculate( threshold ); // calculate tp, fp, tn, fn
		result = static_cast< double >( tn ) / ( tn + fp );
		if ( ( tn + fp ) == 0 ) // check division by zero
			throw DivisionByZero();
	}

	return result;
}

// Calculates and returns the predicitive value positive
double TwoSet::getPVP()
{
	double result = 0;

	if ( !thresholdFlag ) // a threshold must have been set
		util::screen() << "PVP cannot be calculated: a threshold has not been set." << endl;
	else // calculate PVP
	{
		calculate( threshold ); // calculate tp, fp, tn, fn
		result = static_cast< double >( tp ) / ( tp + fp );
		if ( ( tp + fp ) == 0 ) // check division by zero
			throw DivisionByZero();
	}

	return result;
}

// Calculates and returns the predictive value negative
double TwoSet::getPVN()
{
	double result = 0;

	if ( !thresholdFlag ) // a threshold must have been set
		util::screen() << "PVN cannot be calculated: a threshold has not been set." << endl;
	else // calculate PVN
	{
		calculate( threshold ); // calculate tp, fp, tn, fn
		result = static_cast< double >( tn ) / ( tn + fn );
		if ( ( tn + fn ) == 0 ) // check division by zero
			throw DivisionByZero();
	}

	return result;
}

// Confusion-matrix counts at the current threshold. These expose the numbers
//    ClassTable prints; they throw (rather than print-and-return-0 like the
//    rate getters) so a caller building JSON can turn "no threshold" into null.
unsigned TwoSet::getTP()
{
	if ( !thresholdFlag )
		throw twoSetErr( "true positives cannot be counted: no threshold set" );
	calculate( threshold );
	return tp;
}

unsigned TwoSet::getTN()
{
	if ( !thresholdFlag )
		throw twoSetErr( "true negatives cannot be counted: no threshold set" );
	calculate( threshold );
	return tn;
}

unsigned TwoSet::getFP()
{
	if ( !thresholdFlag )
		throw twoSetErr( "false positives cannot be counted: no threshold set" );
	calculate( threshold );
	return fp;
}

unsigned TwoSet::getFN()
{
	if ( !thresholdFlag )
		throw twoSetErr( "false negatives cannot be counted: no threshold set" );
	calculate( threshold );
	return fn;
}

// Outputs to ostream a classification table
void TwoSet::ClassTable( ostream& outputStream )
{
// For formatting integer output in the table
#define OUT12 setw( 12 ) << setfill( ' ' ) << setiosflags( ios::right )

	if ( !thresholdFlag ) // a threshold must have been set
		outputStream << "Classification table can't be constructed: a threshold has not been set." << endl;
	else // output the table
	{
		calculate( threshold ); // calculate tp, fp, tn, fn
		outputStream << "----------------------------------" << endl
			         << "         Predicted T   Predicted F" << endl
			         << "         -----------   -----------" << endl
					 << "Known T:" << OUT12 << tp << "  " << OUT12 << fn << endl
		             << "Known F:" << OUT12 << fp << "  " << OUT12 << tn << endl
                     << "----------------------------------" << endl;
	}

#undef OUT12
}

// Calculates and returns the ROC area by trapezoidal method
double TwoSet::getTrapROCarea()
{
	assert ( nThresholds > 0 ); // must be at least 1

	double rocArea = 0; // initialize the ROC area

	if ( !loadedFlag ) // a Matrix must have been loaded
		util::screen() << "I'm sorry, but a TwoSet Matrix has not yet been defined." << endl;

	else
	{
		// Clear the plot vectors (curve capture ported from the 2005 roc app)
		ROCx.clear();
		ROCy.clear();

		vector< double > test = A.col( 1 ); // extract the test column

		// Find maximum and minimum of test column
		double minimum = *min_element( test.begin(), test.end() ),
			maximum = *max_element( test.begin(), test.end() );

		// Calculate first and incremental value for cycling through thresholds
		double increment = ( maximum - minimum ) / nThresholds;
		double ROCthreshold = minimum + increment;

		// Initial x & y values of ROC curve -- remember it counts backwards!
		double x, y, xOld = 1, yOld = 1;

		// Calculate the ROC area by trapezoids
		for ( unsigned i = 1; i <= nThresholds; i++ )
		{
			calculate( ROCthreshold ); // get tp, fp, tn, fn
			x = static_cast< double >( fp ) / ( tn + fp ); // 1 - specificity
			y = static_cast< double >( tp ) / ( tp + fn ); // sensitivity
			if ( x != xOld ) // if new x
			{
				rocArea += 0.5 * ( yOld + y ) * ( xOld - x ); // calculate area
				xOld = x; // reset x & y values
				yOld = y;

				// For plotting ROC curve
				ROCx.push_back( x );
				ROCy.push_back( y );
			}
			ROCthreshold += increment; // Next threshold
		}
	}

	return rocArea; // return the ROC area
}

// Utility function to calculate true positives, true negatives,
//    false positives and false negatives from TwoSet ADT, given
//    a threshold value
// Takes (double) threshold value as argument
void TwoSet::calculate( const double T )
{
	tp = tn = fp = fn = 0; // Set all counters to zero

	for ( unsigned i = 0; i < n; i++ )
	{
		if ( A( i, 0 ) ) // it's a real positive
		{
			if ( A( i, 1 ) >= T ) // it's predicted positive
				tp++; // it's a true positive

			else // otherwise, it's predicted negative
				fn++; // ...and it's a false negative
		}
		else // it's a real negative
		{
			if ( A( i, 1 ) < T ) // it's predicted negative
				tn++; // It's a true negative

			else // otherwise, it's predicted positive
				fp++; // ...and it's a false positive
		}
	}
}

// Utility method to insure that all real outputs are 0 or 1, takes a Matrix
//    as an argument, returns true if operation was successful
bool TwoSet::checkDiscrete( const Matrix< double >& M ) const
{
	bool success = true; // flag to indicate if operation successful

	for ( unsigned r = 0; r < M.rows(); r++ ) // iterate through data Matrix rows
		if ( M( r, 0 ) != 0 && M( r, 0 ) != 1 ) // check if real output value 0 or 1
		{
			success = false; // if not, set success flag false
			break; // and break out of loop
		}

	return success; // return if operation successful
}

// Statistically calculates and returns the ROC area
// See Wickens, Thomas D., Elementary Signal Detection Theory,
// Oxford University Press, New York, NY 2002, pp. 60-74
// Wickens' binomial error bar on the z scale: the rate p is binomial with
//    variance p(1-p)/N (Eq. 11.2, p. 202), and the delta method carries it
//    through the z transform (Eq. 11.3, p. 202), dividing by the squared normal
//    density at z. Points near 0 or 1 have a tiny density and so a huge error
//    bar -- which is correct, and is exactly the down-weighting the within-bin
//    standard deviation could never express.
static double zErrorBar( double p, unsigned N )
{
	const double INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2*pi)
	double z = stats::invZarea( p );
	double density = INV_SQRT_2PI * exp( -0.5 * z * z );
	return sqrt( p * ( 1 - p ) / N ) / density;
}

double TwoSet::getStatROCarea()
 {
	double rocArea = 0;

	if ( !loadedFlag )
		util::screen() << "I'm sorry, but a TwoSet Matrix has not yet been defined." << endl;
	else
	{
		try
		{
			unsigned n0 = 0, n1 = 0; // class sizes: the binomial denominators
			for ( unsigned row = 0; row < A.rows(); row++ )
				if ( A( row, 0 ) == 0 ) n0++; else n1++;

			vector< double > zH, zF, sH, sF, real = A.col( 1 );
			vector< double >::iterator pR;
			sort( real.begin(), real.end() );
			reverse( real.begin(), real.end() );

			double previous = 0; bool havePrevious = false;
			for ( pR = real.begin(); pR != real.end(); pR++ )
			{
				// One point per DISTINCT threshold. Repeating a tied score's point
				//    once per exemplar does not add information -- it is the same
				//    operating point -- it just replicates it in the fit.
				if ( havePrevious && *pR == previous ) continue;
				previous = *pR; havePrevious = true;

				calculate( *pR );
				double F = static_cast< double >( fp ) / ( tn + fp );
				double H = static_cast< double >( tp ) / ( tp + fn );
				if ( ( F != 0 ) && ( H != 0 ) && ( F != 1 ) && ( H != 1 ) )
				{
					zF.push_back( stats::invZarea( F ) );
					zH.push_back( stats::invZarea( H ) );
					sF.push_back( zErrorBar( F, n0 ) );
					sH.push_back( zErrorBar( H, n1 ) );
				}
			}

			if ( zF.size() < 3 )
				throw twoSetErr( "Too few distinct operating points to fit a zROC line" );

			statPoints = ( unsigned ) zF.size();

			XY line( zF, zH, sF, sH );
			rocArea = stats::Zarea( line.a() / sqrt( 1 + ( line.b() * line.b() ) ) );
			statP = line.q();
			statChi2 = line.chi2();
			binnedFlag = false;
		}
		catch ( stats::statsErr& e ) { throw twoSetErr( e.what() ); }
		catch ( XY::XYErr& e ) { throw twoSetErr( e.what() ); }
	}
	statROCcalcFlag = true;
	return rocArea;
}

// Hanley-McNeil standard error of an empirical (trapezoidal) ROC area A:
//    SE^2 = ( A(1-A) + (n1-1)(Q1-A^2) + (n0-1)(Q2-A^2) ) / (n0*n1),
//    Q1 = A/(2-A), Q2 = 2A^2/(1+A)   [Hanley & McNeil, Radiology 1982]
double TwoSet::hmSE( double A )
{
	// Count the classes from the "known" column
	unsigned n0 = 0, n1 = 0;
	for ( unsigned row = 0; row < this->A.rows(); row++ )
		if ( this->A( row, 0 ) == 0 )
			n0++;
		else
			n1++;

	if ( n0 == 0 || n1 == 0 )
		return 0;

	double Q1 = A / ( 2 - A ), Q2 = 2 * A * A / ( 1 + A );
	double var = ( A * ( 1 - A ) + ( n1 - 1 ) * ( Q1 - A * A )
		+ ( n0 - 1 ) * ( Q2 - A * A ) ) / ( ( double ) n0 * n1 );

	return var > 0 ? sqrt( var ) : 0;
}

// Get Hanley-McNeil standard error of the trapezoidal ROC area
double TwoSet::getTrapSE()
{
	return hmSE( getTrapROCarea() );
}

// Get p value for fitted line, smaller is worse
double TwoSet::getStatP()
{
	if ( !statROCcalcFlag ) // calculate ROC area first
		getStatROCarea();

	return statP;
}

// Get chi-squared value for fitted line
double TwoSet::getStatChi2()
{
	if ( !statROCcalcFlag ) // calculate ROC area first
		getStatROCarea();

	return statChi2;
}

// Count the nonzero, non-one data points -- the ones that survive the z
//    transform in getStatROCarea() and so are available for binning
unsigned TwoSet::countGoodData()
{
	vector< double > real = A.col( 1 ); // extract the "real" column
	vector< double >::iterator pR; // to iterate through real vector
	unsigned goodData = 0; // number of nonzero, non-one data points
	for ( pR = real.begin(); pR != real.end(); pR++ )
	{
		calculate( *pR ); // get tp, fp, tn, fn from vector of "reals"
		double F = static_cast< double >( fp ) / ( tn + fp ); // "false alarm rate", 1 - specificity
		double H = static_cast< double >( tp ) / ( tp + fn ); // "hit rate", sensitivity
		if ( ( F != 0 ) && ( H != 0 ) && ( F != 1 ) && ( H != 1 ) )
			goodData++; // it was a nonzero, non-one data point
	}
	return goodData;
}

// Search the binnings for the best fit p and the best ROC area, caching both.
//    Each binning gives a different Az and a different SE, so the report and
//    any other caller must quote the same fits or they will disagree.
void TwoSet::searchROC( ostream& outputStream )
{
	unsigned goodData = countGoodData();

	searchErrorFlag = false;
	searchErrorMsg.clear();
	bestPfit = ROCfit();
	bestAUCfit = ROCfit();

	if ( maxBins > goodData ) // maximum number of bins cannot exceed data points
	{
		outputStream << "WARNING: Maximum number of bins to be searched (" << maxBins
			<< ") exceeds data (" << goodData << ")," << endl;
		maxBins = goodData / 3;
		outputStream << "Setting Maximum number of bins to " << maxBins << endl;
	}

	nBinFlag = true; // if searching, number of bins must determine bin size

	unsigned oldBins = nBins; // remember original number of bins
	// Do the search
	for ( nBins = minBins; nBins <= maxBins; nBins++ )
	{
		double AUC = 0;
		bool usable = false;
		try { AUC = getStatROCarea(); usable = isfinite( AUC ); }
		catch ( TwoSet::twoSetErr& e ) { searchErrorMsg = e.what(); }

		// One binning failing says nothing about the others, and the stat
		//    members still hold the last successful binning's values -- so
		//    take nothing from a binning that gave no area
		if ( !usable )
			continue;

		if ( !bestAUCfit.valid || AUC > bestAUCfit.az ) // record best ROC area
		{
			bestAUCfit.az = AUC;
			bestAUCfit.chi2 = statChi2;
			bestAUCfit.p = statP;
			bestAUCfit.nBins = nBins;
			bestAUCfit.valid = true;
		}
		// A fit whose p could not be computed is still a fit -- it can win on
		//    area above -- but it cannot be the best-p fit, having no p
		if ( isfinite( statP ) && ( !bestPfit.valid || statP > bestPfit.p ) )
		{
			bestPfit.az = AUC;
			bestPfit.chi2 = statChi2;
			bestPfit.p = statP;
			bestPfit.nBins = nBins;
			bestPfit.valid = true;
		}
	}
	nBins = oldBins; // return number of bins to original

	// The search fails only when no binning at all yielded an area. Any that
	//    did leaves the report, and the bootstrap, with an honest answer.
	searchErrorFlag = !bestAUCfit.valid;
	searchCalcFlag = true;
}

// Percentile interval (and spread) of a set of resampled ROC areas
static TwoSet::CI percentileCI( vector< double >& area, unsigned failures )
{
	TwoSet::CI ci;
	ci.failures = failures;
	ci.resamples = ( unsigned ) area.size();

	if ( area.size() < 20 ) // too few to say anything honest
		return ci;

	sort( area.begin(), area.end() );

	// Linearly interpolated percentile of the sorted areas
	auto pct = [ &area ]( double p )
	{
		double idx = p * ( area.size() - 1 );
		unsigned lo = ( unsigned ) floor( idx ), hi = ( unsigned ) ceil( idx );
		double frac = idx - lo;
		return area[ lo ] * ( 1 - frac ) + area[ hi ] * frac;
	};

	ci.lo = pct( 0.025 );
	ci.hi = pct( 0.975 );

	Population spread( area );
	ci.se = spread.std(); // the bootstrap SD is the standard error

	ci.valid = true;
	return ci;
}

// Bootstrap the binormal ROC areas. Craig's original intent for these
//    intervals (the delta method that stood here was an analytic substitute,
//    and a poor one: its SE tracked the bin count rather than the sample
//    size, because the z-ROC line is fitted to binned points whose error bars
//    measure bin width). Resampling re-runs the whole Wickens procedure and
//    so is indifferent to what happens inside it.
void TwoSet::bootstrapROC()
{
	if ( bootCalcFlag ) // intervals already current
		return;

	bestPci = CI();
	bestAUCci = CI();
	statCi = CI();
	bootCalcFlag = true; // even a refusal below is a current answer

	if ( bootB == 0 || !loadedFlag )
		return;

	// Stratified: resample within each class so every resample keeps the
	//    observed class sizes (an unstratified draw can yield a resample with
	//    no positives at all, which has no ROC)
	vector< unsigned > known0, known1;
	for ( unsigned row = 0; row < A.rows(); row++ )
		if ( A( row, 0 ) == 0 )
			known0.push_back( row );
		else
			known1.push_back( row );

	if ( known0.empty() || known1.empty() ) // single-class data has no ROC
		return;

	vector< double > azBestP, azBestAUC, azStat;
	// Counted apart because a resample can yield an area but no fit p, which
	//    is a failure for the best-p interval only
	unsigned failures = 0, failuresP = 0;

	for ( unsigned b = 0; b < bootB; b++ )
	{
		// Draw with replacement inside each class
		Matrix< double > M( n, 2 );
		unsigned row = 0;
		for ( unsigned i = 0; i < known0.size(); i++, row++ )
		{
			unsigned pick = known0[ util::i_resample( ( unsigned ) known0.size() ) ];
			M( row, 0 ) = A( pick, 0 );
			M( row, 1 ) = A( pick, 1 );
		}
		for ( unsigned i = 0; i < known1.size(); i++, row++ )
		{
			unsigned pick = known1[ util::i_resample( ( unsigned ) known1.size() ) ];
			M( row, 0 ) = A( pick, 0 );
			M( row, 1 ) = A( pick, 1 );
		}

		try
		{
			TwoSet resample( M );
			// The resample must be scored exactly as this set is scored
			resample.setBinSize( binSize );
			resample.setNbins( nBins );
			resample.setNbinsSetsSize( nBinFlag );
			resample.NbinThreshold( binThresh );
			resample.setROCthresh( calcThresh );
			resample.setMinROCSearchBins( minBins );
			resample.setMaxROCSearchBins( maxBins );
			resample.setBootstrapResamples( 0 ); // no bootstrap within a bootstrap

			if ( searchFlag ) // re-run the search: choosing the binning is part
			{                 //    of the procedure, so it is part of the spread
				ostringstream discard; // a resample's warnings are not the user's
				resample.searchROC( discard );
				ROCfit rp = resample.getBestPfit(), ra = resample.getBestAUCfit();
				if ( !ra.valid ) // no binning gave an area: nothing to contribute
				{
					failures++;
					failuresP++;
					continue;
				}
				// Each distribution takes the resamples it has a value for. A
				//    resample with an area but no computable fit p informs the
				//    best-AUC interval even though it cannot inform the best-p one
				if ( rp.valid )
					azBestP.push_back( rp.az );
				else
					failuresP++;
				azBestAUC.push_back( ra.az );
			}
			else
				azStat.push_back( resample.getStatROCarea() );
		}
		catch ( ... ) // a degenerate resample the fit cannot be computed on
		{
			failures++;
			failuresP++;
		}
	}

	if ( searchFlag )
	{
		bestPci = percentileCI( azBestP, failuresP );
		bestAUCci = percentileCI( azBestAUC, failures );
	}
	else
		statCi = percentileCI( azStat, failures );
}

// Search on demand for callers that want the fits without the report. The
//    search only runs when the cache is stale, so asking for these never
//    changes what a later ROCarea() prints.
TwoSet::ROCfit TwoSet::getBestPfit()
{
	if ( !searchCalcFlag )
	{
		ostringstream discard; // the maxBins warning belongs to the report
		searchROC( discard );
	}
	return bestPfit;
}

TwoSet::ROCfit TwoSet::getBestAUCfit()
{
	if ( !searchCalcFlag )
	{
		ostringstream discard;
		searchROC( discard );
	}
	return bestAUCfit;
}

bool TwoSet::getROCsearchFailed()
{
	if ( !searchCalcFlag )
	{
		ostringstream discard;
		searchROC( discard );
	}
	return searchErrorFlag;
}

// Bootstrap on demand, same contract as the fit getters above
TwoSet::CI TwoSet::getBestPci()
{
	bootstrapROC();
	return bestPci;
}

TwoSet::CI TwoSet::getBestAUCci()
{
	bootstrapROC();
	return bestAUCci;
}

TwoSet::CI TwoSet::getStatCi()
{
	bootstrapROC();
	return statCi;
}

// Outputs to ostream an ROC report which if reportFlag is false uses either the
//    trapezoidal method if number of data points < threshold (calcThresh)
//    or statistical method if >= threshold, and uses both if reportFlag is true
//    searchFlag determines if largest p, largest ROC is searched
void TwoSet::ROCarea( ostream& outputStream )
{
	unsigned goodData = countGoodData();

	if ( searchFlag )
		searchROC( outputStream );

	// The intervals the report is about to print
	if ( goodData >= calcThresh )
		bootstrapROC();

	if ( !reportFlag ) // report only one
	{
		if ( goodData >= calcThresh ) // enough data points for statistical calculation
		{
			if ( searchFlag ) // searched for best p, AUC
			{
				outputStream << "Searching for best p:" << endl;
				if ( bestPfit.valid )
					// Use utility method
					statReport( outputStream, goodData, bestPfit.nBins, bestPfit.az, bestPfit.chi2, bestPfit.p, bestPci );
				else if ( bestAUCfit.valid ) // areas, but no fit p to choose among them
					outputStream << "No binning gave a computable fit p value." << endl;
				else
					outputStream << searchErrorMsg << endl;
				outputStream << "Searching for best AUC:" << endl;
				if ( bestAUCfit.valid )
					statReport( outputStream, goodData, bestAUCfit.nBins, bestAUCfit.az, bestAUCfit.chi2, bestAUCfit.p, bestAUCci );
				else
					outputStream << searchErrorMsg << endl;
			}
			else
			{
				double AUC = 0;
				try  // calculate ROC AUC to get chi2, p values
				{
					AUC = getStatROCarea();
					statReport( outputStream, goodData, nBins, AUC, statChi2, statP, statCi ); // use utility method
				}
				catch ( TwoSet::twoSetErr& e )
				{
					outputStream << e.what() << endl;
				}
			}
		}
		else // use trapezoidal method
		{
		outputStream << "ROC area = " << resetiosflags( ios::scientific )
			<< setiosflags( ios::fixed | ios::showpoint ) << setprecision( 6 ) << getTrapROCarea() << endl;
		{
			double A = getTrapROCarea(), se = hmSE( A );
			double lo = A - 1.96 * se, hi = A + 1.96 * se;
			if ( lo < 0 ) lo = 0;
			if ( hi > 1 ) hi = 1;
			outputStream << "95% CI = " << lo << " - " << hi
				<< " (SE = " << se << ", Hanley-McNeil)" << endl;
		}
			outputStream << "WARNING: TRAPEZOIDAL METHOD USED" << endl;
			outputStream << "Number thresholds = " << nThresholds << endl;
			outputStream << resetiosflags( ios::fixed | ios::showpoint );
		}
	}
	else // report both if possible
	{
		if ( goodData >= calcThresh ) // enough data points for statistical calculation
		{
			outputStream << "By statistical method, ";
			if ( searchFlag ) // searched for best p, AUC
			{
				outputStream << "Searching for best p:" << endl;
				if ( bestPfit.valid )
					// Use utility method
					statReport( outputStream, goodData, bestPfit.nBins, bestPfit.az, bestPfit.chi2, bestPfit.p, bestPci );
				else if ( bestAUCfit.valid ) // areas, but no fit p to choose among them
					outputStream << "No binning gave a computable fit p value." << endl;
				else
					outputStream << searchErrorMsg << endl;
				outputStream << "Searching for best AUC:" << endl;
				if ( bestAUCfit.valid )
					statReport( outputStream, goodData, bestAUCfit.nBins, bestAUCfit.az, bestAUCfit.chi2, bestAUCfit.p, bestAUCci );
				else
					outputStream << searchErrorMsg << endl;
			}
			else
			{
				double AUC = 0;
				try  // calculate ROC AUC to get chi2, p values
				{
					AUC = getStatROCarea();
					statReport( outputStream, goodData, nBins, AUC, statChi2, statP, statCi ); // use utility method
				}
				catch ( TwoSet::twoSetErr& e )
				{
					outputStream << e.what() << endl;
				}
			}
		}
		else // not enough data points for statistical calculation
		{
			outputStream << "Cannot calculate ROC statistically, not enough data points ("
				<< calcThresh << ")" << endl;
		}
		outputStream << "By trapezoidal method, ROC area = " << resetiosflags( ios::scientific )
			<< setiosflags( ios::fixed | ios::showpoint ) << setprecision( 6 ) << getTrapROCarea() << endl;
		{
			double A = getTrapROCarea(), se = hmSE( A );
			double lo = A - 1.96 * se, hi = A + 1.96 * se;
			if ( lo < 0 ) lo = 0;
			if ( hi > 1 ) hi = 1;
			outputStream << "95% CI = " << lo << " - " << hi
				<< " (SE = " << se << ", Hanley-McNeil)" << endl;
		}
		outputStream << "Number thresholds = " << nThresholds << endl;
		outputStream << resetiosflags( ios::fixed | ios::showpoint );
	}
}

// Utility method for ROCarea, outputs statistical report
void TwoSet::statReport( ostream& outputStream, unsigned goodData, unsigned nBins, double AUC,
	double chi2, double p, const CI& ci )
{
	outputStream << resetiosflags( ios::scientific ) << setiosflags( ios::fixed | ios::showpoint )
		<< setprecision( 6 );
	outputStream << "ROC area = " << AUC << endl;

	// 95% bootstrap percentile confidence interval
	if ( ci.valid )
	{
		outputStream << "95% CI = " << ci.lo << " - " << ci.hi
			<< " (SE = " << ci.se << ", " << ci.resamples
			<< " bootstrap resamples";
		if ( ci.failures > 0 )
			outputStream << ", " << ci.failures << " failed";
		outputStream << ")" << endl;
	}
	else
		outputStream << "95% CI = not available (too few usable resamples)" << endl;
	// Goodness of fit is a diagnostic and never a precondition for reporting
	//    the area (Wickens, Elementary Signal Detection Theory, section 11.5
	//    p. 217), so an uncomputable one is reported as such and the area above
	//    still stands
	if ( isfinite( chi2 ) )
		outputStream << "Chi-squared = " << chi2 << endl;
	else
		outputStream << "Chi-squared = not available" << endl;
	outputStream << resetiosflags( ios::fixed ) << setiosflags( ios::scientific );
	if ( isfinite( p ) )
		outputStream << "p = " << p << " (closer to 1 is better)" << endl;
	else
		outputStream << "p = not available (fit probability did not converge)" << endl;
	// The zROC line is fitted to the distinct operating points themselves, so
	//    the count of them is what a reader needs -- an Az from five points and
	//    an Az from five hundred are not the same claim. (This line used to
	//    report the bin size and count, or warn that the data "was not binned";
	//    binning is gone, so both were obsolete.)
	outputStream << "Operating points fitted = " << statPoints << endl;
	outputStream << resetiosflags( ios::scientific );
}

// Accessors for Kolmogorov-Smirnov test
double TwoSet::getKSD() // returns D
{
	if ( !KScalcFlag )
		KScalc();

	return KSD;
}

double TwoSet::getKSP() // returns p-value
{
	if ( !KScalcFlag )
		KScalc();

	return KSP;
}

// Utility method to calculate Kolmogorov-Smirnov test Numerical Recipes in C p. 623
// Craig Niederberger modified 3/12/2009 to throw exceptions
void TwoSet::KScalc()
{
	if ( !loadedFlag ) // a Matrix must have been loaded
		util::screen() << "I'm sorry, but a TwoSet Matrix has not yet been defined." << endl;

	else try
	{
		vector< double > zeros, ones;

		for ( unsigned row = 0; row < A.rows(); row++ )
		{
			if ( A( row, 0 ) == 0 )
				zeros.push_back( A( row, 1 ) );
			else
				ones.push_back( A( row, 1 ) );
		}

		sort( zeros.begin(), zeros.end() );
		sort( ones.begin(), ones.end() );

		// Numerical Recipes uses 1-based arrays; these vectors are 0-based.
		// The 2.x code kept NR's 1-based indices, skipping element 0 and
		// reading one past the end (undefined behavior). Fixed for 3.0.
		unsigned long n1 = zeros.size(), n2 = ones.size(), j1 = 0, j2 = 0;
		double d = 0.0, d1, d2, dt, en1 = n1, en2 = n2, fn1 = 0.0, fn2 = 0.0;

		while ( j1 < n1 && j2 < n2 )
		{
			if ( ( d1 = zeros[ j1 ] ) <= ( d2 = ones[ j2 ] ) )
				fn1 = ( double ) ++j1 / en1;
			if ( d2 <= d1 )
				fn2 = ( double ) ++j2 / en2;
			if ( ( dt = fabs( fn2 - fn1 ) ) > d )
				d = dt;
		}

		KSD = d;

		double en = sqrt( en1 * en2 / ( en1 + en2 ) );

		KSP = stats::probks( ( en + 0.12 + 0.11 / en ) * d );

		KScalcFlag = true; // KS test statistics have been calculated
	}
	catch ( stats::statsErr& e ) { throw twoSetErr( e.what() ); }
	catch ( XY::XYErr& e ) { throw twoSetErr( e.what() ); }
}

// Accessors for Pearson's Chi-Square test Hui Liu added 08/15/2004
double TwoSet::getPearsonX2()  // returns p-value
{
	if ( !PKX2calcFlag )
		PKX2calc();

		return PKX2P;
}

// Utility method to calculate Pearson's Chi-Square statistic Hui Liu modified 09/25/2004
// Craig Niederberger modified 3/12/2009 to throw exceptions
void TwoSet::PKX2calc()
{
	if ( !loadedFlag ) // a Matrix must have been loaded
		util::screen() << "I'm sorry, but a TwoSet Matrix has not yet been defined." << endl;

	else try
	{
		double PKX2= 0.0;
		double tmpSumPX= 0.0;

		for ( unsigned iAi1 = 0; iAi1 < n; iAi1++ )
		{
			tmpSumPX += A(iAi1, 1);
		}

		tmpSumPX= tmpSumPX/n;  // Most division_by_zero case happenning is due to A(i,1)=0. So we average A(i,1).

		for ( unsigned i = 0; i < n; i++ )
		{
			if ( ( A ( i, 1 ) * ( 1 - A( i, 1 ) / n ) ) == 0 )
				if ( tmpSumPX > 0.5 )
					PKX2 += ( ( A( i, 0 ) - A( i, 1 ) ) * ( A( i, 0 ) - A( i, 1 ) ) ) / ( tmpSumPX );
				else
					PKX2 += ( A ( i, 0 ) - A( i, 1 ) ) * ( A( i, 0 ) - A( i, 1 ) ) / ( 0.5 );
			else
				PKX2 += ( ( A( i, 0 ) - A( i, 1 ) ) * ( A( i, 0 ) - A( i, 1 ) ) ) / (A( i, 1 ) * ( 1 - A( i, 1 ) / n ) );
		 }


		PKX2P = stats::pX2(2, PKX2 ); // The default degree of freedom for Pearson's Chi-Sqaure is 2
		PKX2calcFlag = true; // Pearson's Chi-Square test statistics have been calculated
	}
	catch ( stats::statsErr& e ) { throw twoSetErr( e.what() ); }
	catch ( XY::XYErr& e ) { throw twoSetErr( e.what() ); }
}

// Accessors for Hosmer-Lemeshow statistic Hui Liu added 08/16/2004
double TwoSet::getHLX2()   // returns p-value
{
	if ( !HLX2calcFlag )
		HLX2calc();

	return HLX2P;
}

// Utility method to calculate Hosmer-Lemeshow statistic
// Referring to the book "Applied Logistic Regression"
// Hui Liu modified 09/25/2004
// Craig Niederberger modified 3/12/2009 to throw exceptions
void TwoSet::HLX2calc()
{
	// Test data is divided into NG (from 10 to 3) groups.
	// Individual group has an unfixed number of test samples.

	unsigned NumGrp[ 10 ];


	double ChiSquarSumHL = 0.0;
	double tmpSumHL = 0.0;
	double tmpHLX2P = 0.0;
	double tempSwap = 0.0;

    double aHL[ 10000 ];
	double bHL[ 10000 ];

    for ( unsigned iM2V = 0; iM2V < n; iM2V++ )
	{
         aHL[ iM2V ] = A( iM2V, 1 );
		 bHL[ iM2V ] = A( iM2V, 0 );
    }

    // Sorting the expected values and observed values from the smallest to largest
    for ( unsigned iSort = 0; iSort < ( n - 1 ); iSort++ )
    {
        unsigned minIndex = iSort;

        // Find the index of the minimum element
        for ( unsigned jSort = iSort + 1; jSort < n; jSort++ )
        {
            if ( aHL[ jSort ] < aHL[ minIndex ] )
            {
                minIndex = jSort;
            }
        }

        // Swap if i-th element not already smallest
        if ( minIndex > iSort )
        {
            tempSwap = aHL[ iSort ];
			aHL[ iSort ] = aHL[ minIndex ];
			aHL[ minIndex ] = tempSwap;

			tempSwap = bHL[ iSort ];
			bHL[ iSort ] = bHL[ minIndex ];
			bHL[ minIndex ] = tempSwap;
        }

    }

	try
	{
		for ( unsigned NG = 10; NG > 3; NG-- )
		{
			int tmpNumSum = 0;

			for ( unsigned iGroup = 0; iGroup < NG-1; iGroup++ )
			{
				double ExpectVal = 0.0;
				double ObserVal = 0.0;

				NumGrp[ iGroup ] = int( n / NG + 0.5); // NG groups for data set

				// Calculate the first (NG-1) groups
				for ( unsigned jGroup = 0; jGroup < NumGrp[ iGroup ]; jGroup++ )
				{
					ExpectVal += aHL[ jGroup+tmpNumSum ];
					ObserVal += bHL[ jGroup+tmpNumSum ];
				}

				tmpNumSum += NumGrp[ iGroup ];

				// Most division_by_zero happening is due to Observal equals to 0.
				double ObserValAvg = ObserVal / NumGrp[ iGroup ];

				tmpSumHL = ObserVal * ( 1.0 - ObserVal / NumGrp[ iGroup ] );

				if ( tmpSumHL != 0 )
					ChiSquarSumHL += ( ( ExpectVal-ObserVal ) * ( ExpectVal-ObserVal ) ) / tmpSumHL;
				else
				{
					if ( ObserValAvg > 0.5 )
						ChiSquarSumHL += ( ( ExpectVal-ObserVal ) * ( ExpectVal-ObserVal ) ) / ObserValAvg;
					else
						ChiSquarSumHL += ( ( ExpectVal-ObserVal ) * ( ExpectVal-ObserVal ) ) / ( 0.5 );
				}

				// Calculate the (NG-1)th group
				NumGrp[ NG - 1 ] = n - tmpNumSum;
				double ExpectValNG = 0.0;
				double ObserValNG = 0.0;
				for ( unsigned jGroupNG = 0; jGroupNG < NumGrp[ NG - 1 ]; jGroupNG++ )
				{
					ExpectValNG += aHL[ jGroupNG + tmpNumSum ];
					ObserValNG += bHL[ jGroupNG + tmpNumSum];
				}

				// Most division_by_zero happening is due to ObservalNG equals to 0.
				double ObserValNGAvg = ObserValNG / NumGrp[ NG - 1 ];

				tmpSumHL = ObserValNG * ( 1.0 - ObserValNG / NumGrp[ NG - 1 ] );

				if ( tmpSumHL != 0.0 )
					ChiSquarSumHL += ( ( ExpectValNG - ObserValNG ) * ( ExpectValNG - ObserValNG ) )
						/ tmpSumHL;
				else
				{
					if ( ObserValNGAvg > 0.5 )
						ChiSquarSumHL += ( ( ExpectValNG - ObserValNG ) * ( ExpectValNG - ObserValNG ) ) / ObserValNGAvg;
					else
						ChiSquarSumHL += ( ExpectValNG - ObserValNG ) * ( ExpectValNG - ObserValNG ) / ( 0.5 );
				}

				// The degree of freedom for Hosmer-Lemeshow is (NG-2)
				HLX2P = stats::pX2( NG-2, ChiSquarSumHL );
			}

			if ( HLX2P > tmpHLX2P ) // chose the largest values from different group number case
				tmpHLX2P = HLX2P;
		}
	}
	catch ( stats::statsErr& e ) { throw twoSetErr( e.what() ); }
	catch ( XY::XYErr& e ) { throw twoSetErr( e.what() ); }

	HLX2calcFlag = true; // Hosmer-Lemeshow test statistics have been calculated
	HLX2P = tmpHLX2P;
}
