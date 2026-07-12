// Header file for TwoSet class, which represents a dataset with
// one variable and one outcome.

#include <string>
#include <algorithm>
#include <stdexcept>
#include <iomanip>

using namespace std;

#include "matrix.h"
#include "vector_ops.h"
#include "stats.h"

// Preprocessor directives to prevent multiple header file inclusion
#ifndef TWOSET_H
#define TWOSET_H

// Tony Makhlouf change 9/12/05: For VC 7.0 ( .NET ), avoid the exception specification warning
#ifdef _MSC_VER
#pragma warning( disable : 4290 )
#endif

// TwoSet ADT (abstract data type) definition
class TwoSet {
public:
	TwoSet(); // default ctor
	TwoSet( const Matrix< double >& D ); // ctor loads Matrix
	~TwoSet(){} // destructor
	TwoSet( const TwoSet& ); // copy constructor
	TwoSet& operator= ( const TwoSet& ); // = operator

	// TwoSet exception, extends runtime_error class
	class twoSetErr : public runtime_error {
	public:
		twoSetErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Division by zero exception, extends runtime_error class
	class DivisionByZero : public runtime_error {
	public:
		DivisionByZero() throw() : runtime_error( "division by zero" ) {}
	};

	// Methods for loading Matrices and files into the TwoSet object
	void setMatrix( const Matrix< double >& ); // loads Matrix into TwoSet object
	void load( string& ); // loads from file
	bool loaded(){ return loadedFlag; } // returns true if Matrix is loaded

	// Accessors for TwoSet elements
	Matrix< double >& getData() { return A; } // returns the dataset Matrix
	bool save( string& ); // save a TwoSet object to file
	unsigned getNumElements() { return n; } // return the number of elements
	TwoSet& insertReal( const vector< double >& ); // insert vector into real column
	vector< double >& real( vector< double >& v ) const; // get vector from real column
	vector< double > real() const; // get new vector from real column
	TwoSet& insertTest( const vector< double >& ); // insert vector into test column
	vector< double >& test( vector< double >& v ) const; // get vector from test column
	vector< double > test() const; // get new vector from test column
	double& real( const unsigned ); // accessor for a row's element in the real column
	const double real( const unsigned ) const; // const accessor for a real element
	double& test( const unsigned ); // accessor for a row element in the test column (invalidates cached stats)
	void invalidate(); // invalidate cached statistics after guesses change
	const double test( const unsigned ) const; // const accessor for a test element

	// Accessors for dataset and calculated metrics
	void setThreshold( const double thresh ); // sets threshold value
	double getClassAcc(); // returns classification accuracy
	double getSens(); // returns sensitivity
	double getSpec(); // returns specificity
	double getPVP(); // returns PVP
	double getPVN(); // returns PVN

	// Outputs to ostream a classification table
	void ClassTable( ostream& );

	// ROC area calculation by trapezoidal method
	// Accessors for number of thresholds to calculate trapezoidal method
	void setTrapThresholds( const unsigned n ) { nThresholds = n; }
	unsigned getTrapThresholds() { return nThresholds; }
	double getTrapROCarea(); // returns ROC by trapezoidal area

	// Accessors for statistical ROC calculation
	void setNbins( const unsigned n ) { nBins = n; } // set number of bins
	unsigned getNbins() { return nBins; } // return number of bins
	void setBinSize( const unsigned n ) { binSize = n; } // set bin size
	unsigned getBinSize() { return binSize; } // return bin size
	// Number of data points under which data will not be binned
	void NbinThreshold( const unsigned n ) { binThresh = n; }
	unsigned getNBinThreshold() { return binThresh; }
	bool getBinned() { return binnedFlag; } // returns if data was binned
	// Accessors to determine if number of bins sets bin size (and vice versa)
	void setNbinsSetsSize( bool flag ) { nBinFlag = flag; }
	bool getNbinsSetsSize() { return nBinFlag; }
	double getStatROCarea(); // returns statistical ROC area
	double getStatP(); // returns p-value for fitted line, smaller is worse
	double getStatChi2(); // returns chi-squared value for fitted line
	double getStatAzSE(); // returns delta-method standard error of Az
	double getTrapSE(); // returns Hanley-McNeil SE of the trapezoidal area

	// Outputs to ostream an ROC report with statistical method and/or
	//    trapezoidal method depending on flag set by setReportFlag() accessor
	void ROCarea( ostream& );
	// Accessors which specify if largest p, largest ROC is searched
	void setROCSearchFlag( const bool flag ) { searchFlag = flag; }
	bool getROCSearchFlag() { return searchFlag; }
	// Access minimum & maximum number of search bins
	void setMinROCSearchBins( const unsigned n ) { assert ( n > 2 ); minBins = n; }
	unsigned getMinROCSearchBins() { return minBins; }
	void setMaxROCSearchBins( const unsigned n ) { assert ( n > 2 ); maxBins = n; }
	unsigned getMaxROCSearchBins() { return maxBins; }
	// Access threshold under which statistics will not be used
	void setROCthresh( const unsigned n ) { calcThresh = n; }
	unsigned getROCthresh() { return calcThresh; }
	// Access flag which if false, report uses either the trapezoidal method
	//    if number of data points < threshold (calcThresh) or statistical
	//    method if >= threshold, and uses both if flag is true
	void setROCReportFlag( const bool flag ) { reportFlag = flag; }
	bool getROCReportFlag() { return reportFlag; }

	// Accessors for Kolmogorov-Smirnov test
	double getKSD(); // returns D
	double getKSP(); // returns p-value

	// Hui Liu added 08/15/2004 Accessors for Pearson's Chi-Square test
	double getPearsonX2();  //;

	// Hui Liu added 08/16/2004 Accessors for Hosmer-Lemeshow test
	double getHLX2();  //;

	private:
	Matrix< double > A; /* Matrix which represents dataset
		format: ('real' = real outcome, 'test' = test prediction)

		real1 test1
		real2 test2
		...
		realN testN

		e.g.

		0.000000 0.002851
		0.000000 0.002978
		0.000000 0.002857
		1.000000 0.571710 */

	unsigned n, // number of elements in array
		tp, // true positives
		tn, // true negatives
		fp, // false positives
		fn, // false negatives
		nThresholds, // number of thresholds for trapezoidal method
		nBins, // number of bins for statistical ROC
		binSize, // bin size for statistical ROC
		binThresh, // number of data points under which data will not be binned
		calcThresh, // number of data points under which statistics will not be used
		minBins, // minimum number of search bins
		maxBins; // maximum number of search bins

	double threshold, // threshold value
		statP, // p-value for fitted line for statistical ROC calculation
		statChi2, // chi-squared value for fitted line for statistical ROC calculation
		statAzSE, // delta-method standard error of the statistical ROC area (Az)
		KSD, // Kolmogorov-Smirnov test D value
		KSP, // Kolmogorov-Smirnov test p-value
	    PKX2P, // Pearson's Chi-Square test p-value   Hui Liu added 08/15/2004
		HLX2P;  // Hosmer-Lemeshow test p-value   Hui Liu added 08/16/2004

	bool loadedFlag, // flag to indicate if Matrix loaded
		thresholdFlag, // flag to indicate if threshold set
		nBinFlag, // flag indicates if number of bins determines bin size
		binnedFlag, // flag indicates if data was binned for statistical ROC calculation
		statROCcalcFlag, // flag indicates if statistical ROC area calculated
		reportFlag, // flag directs how to report statistical and/or trapezoidal ROC
		searchFlag, // flag indicates if largest p, largest ROC is searched
		KScalcFlag, // flag indicates if Kolmogorov-Smirnov test calculated
		PKX2calcFlag,// flag indicates if Pearson's Chi-Square test calculated   Hui Liu added 08/15/2004
		HLX2calcFlag; // flag indicates if Hosmer-Lemeshow test calculated   Hui Liu added 08/16/2004

	// Utility function to set common initial values for constructors
	void initialize();

	// Utility function which calculates tp, tn, fp, fn, takes threshold
	//    as argument
	void calculate( const double );

	// Utility function to check that all real outcomes are discrete
	bool checkDiscrete( const Matrix< double >& ) const;

	// Copy utility
	void copy( const TwoSet& rhs );

	// Utility method for ROCarea, outputs statistical report
	void statReport( ostream&, unsigned, unsigned, double, double, double, double );

	// Delta-method standard error of Az from a fitted zROC line
	double azSE( XY& );

	// Hanley-McNeil standard error of an empirical (trapezoidal) ROC area
	double hmSE( double );

	// Utility method to calculate Kolmogorov-Smirnov test Numerical Recipes in C p. 623
	void KScalc();

	// Utility method to calculate Pearson's Chi-Square test  Hui Liu added 08/16/2004
	void PKX2calc();

	// Utility method to calculate Hosmer-Lemeshow test  Hui Liu added 08/17/2004
	void HLX2calc();
};

#endif
