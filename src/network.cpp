// Header file for Network, the abstract base class for neUROn2++ neural networks
//    which use some form of iterative training method.
//    All Networks have weights in some form which may be randomized,
//    a feed forward method, a training method, and a learning rate

#include "stdafx.h" // For MSVC, must be first!

#include <gsl/gsl_eigen.h>  // GSL eigenvalue routines
#include <limits>           // quiet_NaN: a degenerate information matrix has none

#include "network.h"

// Default constructor
Network::Network()
{
	builtFlag = false; // the Network object hasn't yet been built
	weightsSetFlag = false; // weights not set yet
	finalFlag = false; // it isn't the last training iteration by default

	// Good ideas for initial values
	randomLimit = 0.5; // a good initial random limit
	biasFlag = true; // Networks should generally have bias nodes
	batchEpochFlag = true; // Networks should generally use batch/epoch training
	weightDecayFlag = true; // Networks should generally use weight decay
	// Weight-decay strength. The L2 penalty added to the error is
	//    lambda*sum(w^2) = (1/(2*sigma^2))*sum(w^2) -- a zero-mean Gaussian prior
	//    N(0, sigma^2) on the weights, sigma being how large a weight is expected
	//    a priori (large sigma = weak decay). The value STORED here is 2*lambda =
	//    1/sigma^2 (the code reads it as 2*lambda: runHeader sets regularizer =
	//    decay/2, and the gradient of the penalty is decay*w). So 5e-5 is a
	//    deliberately weak default -- weights are only nudged, not biased.
	//    Historical wrinkle: the value was chosen as lambda for sigma=100
	//    (1/(2*100^2) = 5e-5) but stored where 2*lambda belongs, so the EFFECTIVE
	//    sigma is 1/sqrt(5e-5) ~ 141, not 100. Left as-is: it is oracle/golden-
	//    pinned and the difference is immaterial for so weak a prior.
	decay = 5e-5;

	// Constants for automatic stepsize selection
	automaticStepSizeFlag = false; // doesn't start with automatic stepsize
	deltaError = 1e-15;
	gamma = 0.8;
	maxLoops = 3;

	// Values that really should be set in the driver
	// The learning rate should be set in the driver to be 1 or 1/N depending on
	//    if training is off-line (batch/epoch) or on-line
	eta = 0.05;

	// The DERIVED decay constants, set here as defence only -- a caller that
	//    reaches innerTrainSet() without going through train() must not read
	//    indeterminate storage. prepareRun() remains the authoritative
	//    calculation, because eta and decay may both change between
	//    construction and a run. Deliberately placed AFTER eta and decay are
	//    assigned: derived state must be computed from initialized inputs, and
	//    putting it above eta would reproduce, in the fix, the very defect
	//    being fixed. Explicitly qualified: a virtual call from a constructor
	//    dispatches to this class anyway, and saying so removes the question.
	Network::prepareRun();

	// Start with canonical backprop
	trainingType = 0;

	// Condition-number diagnostics not computed yet
	condNum = -1;
	condMaxEig = -1;
	condMinEig = -1;
	currGradMax = 0;
}

// Default destructor
Network::~Network() { }

// Copy constructor
Network::Network( const Network& rhs )
{
	Network::copy( rhs ); // use the copy utility
}

// Overloaded = operator
Network& Network::operator= ( const Network& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		Network::copy( rhs ); // use the copy utility

	return *this; // enables A = B = C
}

// Copy utility
void Network::copy( const Network& rhs )
{
	Iterative::copy( rhs ); // call immediate base object copy
	output = rhs.output;
	xs = rhs.xs;
	o = rhs.o;
	x = rhs.x;
	randomLimit = rhs.randomLimit;
	eta = rhs.eta;
	decay = rhs.decay;
	decayTerm = rhs.decayTerm;
	regularizer = rhs.regularizer;
	weightsSetFlag = rhs.weightsSetFlag;
	biasFlag = rhs.biasFlag;
	batchEpochFlag = rhs.batchEpochFlag;
	weightDecayFlag = rhs.weightDecayFlag;
	trainingType = rhs.trainingType;
	stackG = rhs.stackG;
	lastF = rhs.lastF;
	lastG = rhs.lastG;
	o_errAccumulate = rhs.o_errAccumulate;
	oldErrorAccumulate = rhs.oldErrorAccumulate;
	automaticStepSizeFlag = rhs.automaticStepSizeFlag;
	deltaError = rhs.deltaError;
	gamma = rhs.gamma;
	maxLoops = rhs.maxLoops;
	finalFlag = rhs.finalFlag;
	condNum = rhs.condNum;
	condMaxEig = rhs.condMaxEig;
	condMinEig = rhs.condMinEig;
	currGradMax = rhs.currGradMax;
}

// Set the learning rate
void Network::setEta( const double n )
{
	assert( n >= 0 && n <= 1 );
	eta = n;
}

// Set the weight decay value
void Network::setDecay( const double n )
{
	assert( n >= 0 && n <= 1 );
	decay = n;
}

// Outputs to ostream reporting the accuracy of a Network Model
void Network::reportAccuracy( ostream& outputStream )
{
	unsigned r, // row iterator
		nInput = theData.getInput(); // number of inputs (easier on the eyes)

	double setError = 0, // set error for test & training sets
		accuracy = 0; // classification accuracy for multiple output sets

	if ( theData.getOutput() == 1 ) // accuracy report for 1-output Networks
	{
		if ( theData.trainLoaded() ) // final training set error
		{
			// Iterate through training exemplars
			for ( r = 0; r < Train.rows(); r++ )
			{
				// Forward propagate the training exemplar
				forward( Train, r );

				// Calculate error for single output and add to set error
				errorFunction E( theData.getTrainMatrix()( r, nInput ), o, x, errorType );
				setError += E.value();

				// If discrete, populate TwoSet object for accuracy report
				if ( theData.getDiscrete() && theData.getTrainTwoSet().loaded() )
					theData.getTrainTwoSet().test( r ) = o;
			}

			// Send resulting report to output stream
			outputStream << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific ) << setprecision( 6 )
				<< "Final " << errorLabel << " error in the training set = "
				<< setError / Train.rows() << "." << endl;

			// If cross-entropy, report log likelihood
			if ( errorType == 1 )
				outputStream << "Log likelihood = -" << setError << endl;
		}

		if ( theData.testLoaded() ) // final test set error
		{
			setError = 0; // reset set error

			// Iterate through test exemplars
			for ( r = 0; r < Test.rows(); r++ )
			{
				// Forward propagate the test exemplar
				forward( Test, r );

				// Calculate error for single output and add to set error
				errorFunction E( theData.getTestMatrix()( r, nInput ), o, x, errorType );
				setError += E.value();

				// If discrete, populate TwoSet object for accuracy report
				if ( theData.getDiscrete() && theData.getTestTwoSet().loaded() )
					theData.getTestTwoSet().test( r ) = o;
			}

			// Send resulting report to output stream
			outputStream << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific ) << setprecision( 6 )
				<< "Final " << errorLabel << " error in the test set = "
				<< setError / Test.rows() << "." << endl;

			// If cross-entropy, report log likelihood
			if ( errorType == 1 )
				outputStream << "Log likelihood = -" << setError << endl;
		}

		if ( theData.getDiscrete() ) // output TwoSet metrics report if discrete
			theData.metricsReport( outputStream );
	}

	else // multiple outputs
	{
		if ( theData.trainLoaded() )
		{
			setError = 0; // reset set error

			// Iterate through training exemplars
			for ( r = 0; r < Train.rows(); r++ )
			{
				// Forward propagate the training exemplar
				forward( Train, r );

				// Calculate error for multiple output and add to set error
				errorFunction E( TrainOutput.row( r ), output, xs, errorType );
				setError += E.value();
			}

			// Send resulting report to output stream
			outputStream << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific ) << setprecision( 6 )
				<< "Final " << errorLabel << " error in the training set = "
				<< setError / Train.rows() << "." << endl;

			// If cross-entropy, report log likelihood
			if ( errorType == 1 )
				outputStream << "Log likelihood = -" << setError << endl;

			// If descrete, report final accuracy
			if ( theData.getDiscrete() )
			{
				accuracy = 0; // reset the accuracy

				for ( r = 0; r < Train.rows(); r++ ) // iterate through Train Matrix rows
				{
					forward( Train, r ); // calculate the network's output
					// Make output discrete
					func( output, threshold( theData.getThreshold() ), output );
					if ( output == TrainOutput.row( r ) ) // compare known to guess
						accuracy++;
				}

				accuracy /= Train.rows(); // calculate the accuracy

				// Send formatted training classification accuracy to ostream
				outputStream << resetiosflags( ios::scientific )
					<< setiosflags( ios::fixed ) << setprecision( 1 )
					<< "Final classification accuracy in the training set = "
					<< accuracy * 100 << "." << endl;
			}
		}

		if ( theData.testLoaded() ) // for test set
		{
			setError = 0; // reset set error

			// Iterate through test exemplars
			for ( r = 0; r < Test.rows(); r++ )
			{
				// Forward propagate the test exemplar
				forward( Test, r );

				// Calculate error for multiple output and add to set error
				errorFunction E( TestOutput.row( r ), output, xs, errorType );
				setError += E.value();
			}

			// Send resulting report to output stream
			outputStream << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific ) << setprecision( 6 )
				<< "Final " << errorLabel << " error in the test set = "
				<< setError / Test.rows() << "." << endl;

			// If cross-entropy, report log likelihood
			if ( errorType == 1 )
				outputStream << "Log likelihood = -" << setError << endl;

			// If descrete, report final accuracy
			if ( theData.getDiscrete() )
			{
				accuracy = 0; // reset the accuracy

				for ( r = 0; r < Test.rows(); r++ ) // iterate through Test Matrix rows
				{
					forward( Test, r ); // calculate the network's output
					// Make output discrete
					func( output, threshold( theData.getThreshold() ), output );
					if ( output == TestOutput.row( r ) ) // compare known to guess
						accuracy++;
				}

				accuracy /= Test.rows(); // calculate the accuracy

				// Send formatted training classification accuracy to ostream
				outputStream << resetiosflags( ios::scientific )
					<< setiosflags( ios::fixed ) << setprecision( 1 )
					<< "Final classification accuracy in the test set = "
					<< accuracy * 100 << "." << endl;
			}
		}
	}
}

// Mean error over every stride-th test exemplar (the GUI's mid-training
//    test-error sample). Mirrors reportAccuracy's 1-output test loop but
//    deliberately does NOT write the TwoSet guesses: a sample taken mid-run
//    must never invalidate cached statistics or masquerade as a final pass.
double Network::sampleTestError( unsigned stride )
{
	// The held-out MONITOR (OBD's architecture-selection early-stopping and the
	//    GUI's realtime chart). Phase 4c: sample the VALIDATION set when one is
	//    loaded, so model/architecture selection never touches the test set (the
	//    no-leakage invariant); with no validation set it falls back to the test
	//    set -- the pre-4c behavior, so every existing run is unchanged.
	//    DataSet::monitorSet() IS that rule; asking it here is what keeps the
	//    sampler and every label that names the monitored set in agreement.
	DataSet::MonitorSet which = theData.monitorSet();
	if ( which == DataSet::MONITOR_NONE )
		return -1; // nothing to sample
	bool useVal = ( which == DataSet::MONITOR_VALIDATION );

	if ( stride < 1 )
		stride = 1;

	unsigned nInput = theData.getInput(), count = 0;
	double setError = 0;

	Matrix< double >& X = useVal ? Validation : Test;
	Matrix< double >& M = useVal ? theData.getValMatrix() : theData.getTestMatrix();

	for ( unsigned r = 0; r < X.rows(); r += stride )
	{
		forward( X, r ); // forward propagate the held-out exemplar

		// Same error the final report uses, averaged over the sample
		errorFunction E( M( r, nInput ), o, x, errorType );
		setError += E.value();
		count++;
	}

	return count ? setError / count : -1;
}

// Outputs classification accuracies to ostream for Iterative table
//    Iterative.cpp has already checked that outputs are discrete
void Network::classAccuracy( ostream& outputStream )
{
	unsigned r; // row iterator

	if ( theData.getOutput() == 1 ) // for 1-output Networks
	{
		// Training TwoSet must exist
		if ( theData.trainLoaded() && theData.getTrainTwoSet().loaded() )
		{
			for ( r = 0; r < Train.rows(); r++ ) // iterate through Train Matrix rows
			{
				forward( Train, r ); // calculate the network's output
				// Set the DataSet training TwoSet member's test member to the output
				theData.getTrainTwoSet().test( r ) = o;
			}

			// Send formatted training classification accuracy to ostream
			outputStream << resetiosflags( ios::scientific )
				<< setiosflags( ios::fixed );
			outputStream << "  " << setw( 10 ) << setfill( ' ' ) << setprecision( 1 )
				<< theData.getTrainTwoSet().getClassAcc() * 100;
		}

		// Test TwoSet must exist
		if ( theData.testLoaded() && theData.getTestTwoSet().loaded() )
		{
			for ( r = 0; r < Test.rows(); r++ ) // iterate through Test Matrix rows
			{
				forward( Test, r ); // calculate the network's output
				// Set the DataSet test TwoSet member's test member to the output
				theData.getTestTwoSet().test( r ) = o;
			}

			// Send formatted training classification accuracy to ostream
			outputStream << resetiosflags( ios::scientific )
				<< setiosflags( ios::fixed );
			outputStream << "  " << setw( 10 ) << setfill( ' ' ) << setprecision( 1 )
				<< theData.getTestTwoSet().getClassAcc() * 100;
		}
	}

	else // multiple outputs
	{
		double accuracy; // the classification accuracy

		// For the training set
		if ( theData.trainLoaded() )
		{
			accuracy = 0; // reset the accuracy

			for ( r = 0; r < Train.rows(); r++ ) // iterate through Train Matrix rows
			{
				forward( Train, r ); // calculate the network's output
				// Make output discrete
				func( output, threshold( theData.getThreshold() ), output );
				if ( output == TrainOutput.row( r ) ) // compare known to guess
					accuracy++;
			}

			accuracy /= Train.rows(); // calculate the accuracy

			// Send formatted training classification accuracy to ostream
			outputStream << resetiosflags( ios::scientific )
				<< setiosflags( ios::fixed );
			outputStream << "  " << setw( 10 ) << setfill( ' ' ) << setprecision( 1 )
				<< accuracy * 100;
		}

		// For the test set
		if ( theData.testLoaded() )
		{
			accuracy = 0; // reset the accuracy

			for ( r = 0; r < Test.rows(); r++ ) // iterate through Test Matrix rows
			{
				forward( Test, r ); // calculate the network's output
				// Make output discrete
				func( output, threshold( theData.getThreshold() ), output );
				if ( output == TestOutput.row( r ) ) // compare known to guess
					accuracy++;
			}

			accuracy /= Test.rows(); // calculate the accuracy

			// Send formatted test classification accuracy to ostream
			outputStream << resetiosflags( ios::scientific )
				<< setiosflags( ios::fixed );
			outputStream << "  " << setw( 10 ) << setfill( ' ' ) << setprecision( 1 )
				<< accuracy * 100;
		}
	}
}

// Derive the per-run weight-decay constants. Called by Iterative::train()
//    before the training loop and outside every reporting guard.
//
//    These two statements used to open runHeader(), which train() called only
//    when the run was NOT quiet -- so a quiet run (Network's default has weight
//    decay ON) multiplied its weights by an uninitialised decayTerm and added an
//    uninitialised regularizer to its reported error. Neither constructor set
//    them. Reachable by a fresh model trained quietly or a clone of an untrained
//    template; NOT by the known stepwise path, whose candidates clone an
//    already-trained model and inherit both constants through Network::copy.
//    Latent undefined behaviour rather than a corrupted shipped result. Measured
//    2026-08-01: NaN under -ftrivial-auto-var-init=pattern, and a stable but
//    wrong 0.673 instead of 0.0223 under an ordinary Release build -- the same
//    source and seed giving a different fit depending on the stack.
//    See tests/iterative/check_quietprep.cpp.
void Network::prepareRun()
{
	regularizer = decay / 2; // the regularization term lambda
	decayTerm = 1 - ( eta * decay ); // 1 - 2*eta*lambda
}

// Utility to output Network specific parameters prior to an Iterative run.
//    REPORTING ONLY -- the decay constants it used to derive now live in
//    prepareRun(), because the training math reads them.
void Network::runHeader( ostream& outputStream )
{
	switch ( trainingType ) // the training algorithm
	{
		case 0:
			outputStream << "Training algorithm is canonical backpropagation" << endl;
			break;
		case 1:
			outputStream << "Training algorithm is conjugate gradient descent" << endl;
			break;
		case 2:
			outputStream << "Training algorithm is Shanno" << endl;
			break;
	}

	if ( batchEpochFlag ) // batch/epoch learning
		outputStream << "Training is off-line (batch/epoch is on)" << endl;
	else
		outputStream << "Training is on-line (batch/epoch is off)" << endl;

	if ( automaticStepSizeFlag ) // automatic stepsize selection
		outputStream << "Automatic stepsize selection is on" << endl;
	else // report out eta
		outputStream << "Learning rate eta: " << eta << endl;

	if ( weightDecayFlag ) // weight decay
		outputStream << "Weight decay term lambda: " << decay << endl;
	else
		outputStream << "Weight decay is off" << endl;
}

// Returns maximum gradient of this Network object
double Network::getGradMax()
{
	// If stackG is not created in trainSet() (e.g. canonical backprop),
	//    then one must be created.
	// NOTE: this implies that all other types of Networks create stackG.
	//    If you are making a Network that does not create stackG, you must
	//    change this if statement so that it will be packed.
	if ( trainingType == 0 )
	{	
		pack();
		return maxabs( stackG );
	}

	return currGradMax;
}

// The master training engine, takes training type and iteration as arguments,
//    calls the appropriate training algorithm
void Network::engine( unsigned type, unsigned t )
{
	switch ( type ) // choose training algorithm
	{
		case 1:
			CGD( t ); // conjugate gradient descent
			break;
		case 2:
			shanno( t ); // Shanno algorithm
			break;
	}
}

// Congugate gradient descent, Golden pp. 221-222
void Network::CGD( unsigned t )
{
	pack(); // creates gradient vector stackG from gradient structure
	currGradMax = maxabs( stackG ); // get maximum gradient for report

	if ( ( t == 0 ) || ( t == df() ) ) // Golden, step 1
		// $\vec{f}(t)=-\vec{g}_t$
		// Note that lastF refers to *negative* F in Golden's book,
		// So for the purposes here, return stackG,
		// and remember $\vec{g}_{t-1}$ for next iteration
		lastG = stackG;

	else // Golden, step 2
	{
		// $\vec{u}_t=\vec{g}_t-\vec{g}_{t-1}$
		vector< double > u = stackG - lastG;
		// Remember $\vec{g}_{t-1}$ for next iteration
		lastG = stackG;

		// Since lastF refers to *negative* F in Golden's book
		// D will thus have reversed sign (-) from Golden's book
		double D = dotprod( u, lastF );

		if ( D != 0 ) // catch division by zero
		{
			// $b_t=\frac{\vec{u}^T\vec{g_t}}{\vec{u}^T_t\vec{f}(t-1)}$,
			// $b_t$ will thus have reversed sign (-) from Golden's book
			double b = dotprod( u, stackG ) / D;
			// $\vec{f}(t)=-\vec{g}_t+b_t\vec{f}(t-1)$
			// with the sign notes from above
			stackG -= ( lastF * b );
		}
	}

	// stackG was loaded with $\vec{f}$
	lastF = stackG;

	unpack(); // return gradient structure from stackG
}

// Shanno algorithm, Golden pp. 217-218
void Network::shanno( unsigned t )
{
	pack(); // creates gradient vector stackG from gradient structure
	currGradMax = maxabs( stackG ); // get maximum gradient for report

	if ( ( t == 0 ) || ( t == df() ) ) // Golden, step 1
		// $\vec{f}(t)=-\vec{g}_t$
		// Note that lastF refers to *negative* F in Golden's book,
		// So for the purposes here, return stackG,
		// and remember $\vec{g}_{t-1}$ for next iteration
		lastG = stackG;

	else // Golden, step 2
	{
		// $\vec{u}_t=\vec{g}_t-\vec{g}_{t-1}$
		vector< double > u = stackG - lastG;
		// Remember $\vec{g}_{t-1}$ for next iteration
		lastG = stackG;

		// Since lastF refers to *negative* F in Golden's book
		// D will thus have reversed sign (-) from Golden's book
		double D = dotprod( lastF, u ); // denominator for all coefficients

		if ( D != 0 ) // catch division by zero
		{
			// $a_t=\frac{\vec{f}(t-1)^T\vec{g_t}}{\vec{f}(t-1)^T\vec{u}_t}$,
			// $a_t$ will thus have same sign as Golden's book
			double a = dotprod( lastF, stackG ) / D;
			// $b_t=\frac{\vec{u}^T\vec{g_t}}{\vec{f}(t-1)^T\vec{u}_t}$,
			// $b_t$ will thus have reversed sign (-) from Golden's book
			double b = dotprod( u, stackG ) / D;
			// $c_t=\gamma+\frac{|\vec{u}_t|^2}{\vec{f}(t-1)^T\vec{u}_t}$,
			// 2nd term in $c_t$ will have reversed sign (-) from Golden's book
			double c = eta - ( squared( u ) / D );

			// $\vec{f}(t)=-\vec{g}_t+a_t\vec{u}_t+(b_t-c_ta_t)\vec{f}(t-1)$
			// with the sign notes from above
			stackG -= ( u * a );
			lastF *= ( b  + ( c * a ) );
			stackG -= lastF;
		}
	}

	// stackG was loaded with $\vec{f}$
	lastF = stackG;

	unpack(); // return gradient structure from stackG
}

// These are the old, inefficient forms of these algorithms--but they may be easier
// to read and understand, so don't uncomment them
/* / Congugate gradient descent, Golden pp. 221-222
void Network::CGD( unsigned t )
{
	pack(); // creates gradient vector stackG from gradient structure
	vector< double > F, G = stackG;
	currGradMax = maxabs( G ); 

	if ( ( t == 0 ) || ( t == df() ) ) // Golden, step 1
	{
		F = G * -1.0;
		lastG = G;
		lastF = F;
	}

	else // Golden, step 2
	{
		vector< double > u = G - lastG;
		lastG = G;

		double b = dotprod( u, G ) / dotprod( u, lastF );
		F = ( G * -1.0 ) + ( lastF * b );

		lastF = F;
	}

	stackG = F * -1.0; // because later on, by Methodology section's signs,
	                   // $W = W - \eta G$ (note - sign)
	unpack();
}

// Shanno algorithm, Golden pp. 217-218
void Network::shanno( unsigned t )
{
	pack(); // creates gradient vector stackG from gradient structure
	vector< double > F, G = stackG;
	currGradMax = maxabs( G ); 

	if ( ( t == 0 ) || ( t == df() ) ) // Golden, step 1
	{
		F = G * -1.0;
		lastG = G;
		lastF = F;
	}

	else // Golden, step 2
	{
		vector< double > u = G - lastG;
		lastG = G;

		double D = dotprod( lastF, u );
		double a = dotprod( lastF, G ) / D;
		double b = dotprod( u, G ) / D;
		double c = eta + ( squared( u ) / D );
		F = ( G * -1.0 ) + ( u * a ) + ( lastF * ( b - ( c * a ) ) );

		lastF = F;
	}

	stackG = F * -1.0; // because later on, by Methodology section's signs,
	                   // $W = W - \eta G$ (note - sign)
	unpack();
} */

// Store the eigenvalue diagnostics of a symmetric matrix: largest and smallest
//    absolute eigenvalue, and their ratio. reportCondNum prints these.
//
//    The matrix comes from the caller. Logistic passes X'VX, the observed Fisher
//    information of the UNPENALIZED log likelihood -- the curvature of the
//    design, which is what a collinearity diagnostic must measure. It used to be
//    built here as G G^T / N from per-exemplar gradients harvested by running a
//    training iteration, which (a) trained the model as a side effect of
//    reporting, (b) stored conjugate DIRECTIONS rather than gradients under CGD
//    and Shanno, (c) read an unwritten Matrix under canonical backprop with
//    gradient stopping off, and (d) included the weight-decay penalty, so
//    regularization improved the reported conditioning and could hide
//    collinearity. All four are gone with the change of definition.
void Network::conditionOf( const Matrix< double >& symmetric )
{
	unsigned dimension = symmetric.rows();

	// The contract is a SQUARE symmetric matrix, and gsl_matrix_view_array
	//    below reads dimension x dimension doubles from the storage it is
	//    handed. A non-square argument would make it read past the end -- an
	//    unconditional check, not an assert, because asserts vanish under
	//    NDEBUG and this is a bounds question (standing rule 4's argument for
	//    Matrix::operator() throwing in release builds).
	if ( symmetric.cols() != dimension )
		throw Matrix< double >::BoundsViolation();

	// A model with no estimated parameters has no information matrix and
	//    therefore no condition number. Not hypothetical: forward stepwise
	//    regression begins by training a BASELINE network with every input
	//    removed, and on a logistic model that baseline reaches here with
	//    dimension 0 -- gsl_vector_alloc( 0 ) yields nothing usable and the
	//    eigenvalue read that followed dereferenced null, segfaulting the whole
	//    process, the GUI server included (legacy bug #11).
	//    dimension 1 needs no special case: the loop below copies its single
	//    eigenvalue and the ratio is 1, which is the right answer for a
	//    one-parameter model (and NaN if that eigenvalue is 0, a singular fit).
	if ( dimension == 0 )
	{
		condMaxEig = condMinEig = condNum
			= numeric_limits< double >::quiet_NaN();
		return; // not a number, and reported as such (jnum -> JSON null)
	}

	// GSL wants a contiguous double*; copy so the caller's matrix is untouched
	//    (gsl_eigen_symm overwrites the matrix it is given)
	Matrix< double > work( symmetric );

	gsl_matrix_view m = gsl_matrix_view_array( work.begin(), dimension, dimension );

	gsl_vector *eval = gsl_vector_alloc( dimension );
	gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc( dimension );

	gsl_eigen_symm( &m.matrix, eval, w );

	// Copy out ALL `dimension` eigenvalues.
	//
	//    This was built as the iterator range [ &e[0], &e[dimension-1] ) --
	//    half-open, ending AT the last eigenvalue, so it held dimension-1 of
	//    them and the last was silently discarded from every condition number
	//    the engine ever reported. At dimension 2 the effect is total: one
	//    eigenvalue survives, maximum and minimum become the same number, and
	//    the condition number is exactly 1 -- a perfectly conditioned model,
	//    reported for data that is nothing of the kind.
	//
	//    An indexed copy rather than a corrected pointer range: forming an
	//    iterator one past the end of a GSL vector means reaching for an
	//    address the library does not promise, to save a loop that costs
	//    nothing at these sizes.
	vector< double > eigenv;
	eigenv.reserve( dimension );
	for ( unsigned i = 0; i < dimension; i++ )
		eigenv.push_back( gsl_vector_get( eval, i ) );

	gsl_eigen_symm_free( w );
	gsl_vector_free( eval );

	condMaxEig = maxabs( eigenv );
	condMinEig = minabs( eigenv );
	condNum = condMaxEig / condMinEig;
}

// Utility to report out the condition number, passed argument is the output
//    string
void Network::reportCondNum( ostream& outputStream )
{
	// Prints what conditionOf() stored; it does NOT compute. Deriving a
	//    diagnostic inside a print routine is how reporting came to train the
	//    model in the first place.

	// A degenerate information matrix (a model with no estimated parameters,
	//    e.g. forward
	//    stepwise's empty baseline network) has no eigenvalues to report. Say so
	//    rather than printing "nan" three times.
	if ( !std::isfinite( condNum ) )
	{
		outputStream << "Condition number = not available (the model has no "
			<< "estimated parameters, or its information matrix is singular)" << endl;
		return;
	}

	// Output results
	// outputStream << "Eigenvalues are: " << eigenv << endl; // for debugging
	outputStream << "Information matrix maximum eigenvalue = " << condMaxEig << endl
		         << "                    minimum eigenvalue = " << condMinEig << endl
		<< "Condition number = " << condNum << endl;
}
