// Methods shared by the two 1 output node, 1 hidden layer backpropagation
// networks, SimpleProp (with biases) and BareProp (without). See onehidden.h
// for what stays in the concrete classes and why.

#include "stdafx.h" // For MSVC, must be first!

#include "onehidden.h"

// Copy utility
void OneHiddenNet::copy( const OneHiddenNet& rhs )
{
	Network::copy( rhs ); // call immediate base object copy
	nHidden = rhs.nHidden;
	hW = rhs.hW;
	hWup = rhs.hWup;
	y = rhs.y;
	I = rhs.I;
	hO = rhs.hO;
	oW = rhs.oW;
	h_err = rhs.h_err;
	o_err = rhs.o_err;
	hG = rhs.hG;
	oG = rhs.oG;
}

// Randomizes initial weights in this Network object
void OneHiddenNet::randomize()
{
	assert( builtFlag ); // the object must have been built first

	// Randomize the weights
	hW.random( randomLimit ); // randomize hW
	nvec::random( oW, randomLimit ); // randomize oW

	// Log to history file
	if ( historyFlag ) // make sure flag for history is set
	{
		// Construct the message to the output stream
		ostringstream fileStream;
		fileStream << endl << "I've randomized the weights of a " << objType << " object."
			<< endl << endl;
		addHistory( fileStream ); // append to history file
	}

	weightsSetFlag = true; // set flag to indicate weights now set
}

// Save this Network object to a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully saved
bool OneHiddenNet::save( string& filename )
{
	bool success = false; // flag to indicate file successfully saved

	// Open the output file to save data, overwrite file if it exists
	ofstream savefile( filename.c_str(), ios::out | ios::trunc );

	// Test to insure it was opened
	if ( !savefile.is_open() )
		util::screen() << "Error in opening file to save " << objType << " network!" << endl;
	else
	{
		outputHeader( savefile ); // output the header to the file

		// Next line is hidden layer label, followed by hidden layer matrix
		savefile << "Hidden layer:" << endl;
		savefile << hW.setHeader( false ); // no header is necessary

		// Followed by output layer label, followed by output layer vector
		savefile << "Output layer:" << endl;
		savefile << oW << endl;

		// Print message to user notifying successful save to file
		util::screen() << "The " << objType << " network was successfully saved to " << filename;
		util::screen() << "." << endl;

		savefile.close(); // close output file

		success = true; // set flag to indicate file successfully saved

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the output stream with object name and header
			ostringstream fileStream;
			fileStream << "I've saved the file " << filename << ":" << endl;
			outputHeader( fileStream ); // output the header to the history file
			fileStream << endl;

			addHistory( fileStream ); // append to history file
		}
	}

	return success; // return flag to indicate if file successfully saved
}

// Load this Network object from a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully loaded
bool OneHiddenNet::load( string& filename )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes

	bool success = false; // flag to indicate file successfully loaded

	string lineString, // holder for strings pulled from file
		goodFile; // the name of the actual file

	// Open the file as read-only, and associate it with the 'label' loadfile
	goodFile = util::getGoodFile( filename );
	ifstream loadfile( goodFile.c_str(), ios::in );

	getline( loadfile, lineString ); // 1st line should be this object's type
	util::chopEndl( lineString ); // remove <cr> from end of string if exists
	assert( lineString == objType );

	// The bias line the concrete class's outputHeader wrote: "Bias nodes on all
	//    layers by definition" or "No bias nodes by definition". It is read past
	//    rather than read: the file cannot tell this object what it is, because
	//    the type on the line above already did.
	getline( loadfile, lineString );

	unsigned nInputFromFile; // number of input nodes obtained from file
	loadfile >> nInputFromFile; // retrieve number of input nodes
	getline( loadfile, lineString ); // " input nodes"

	// Make sure number of input nodes matches dataset
	if ( nInputFromFile != nInput )
	{
		util::screen() << "I cannot load this file:" << endl;
		util::screen() << "The number of input nodes do not match the dataset ( ";
		util::screen() << nInput << " )" << endl;
	}
	else
	{
		getline( loadfile, lineString ); // "1 hidden layer by definition"

		loadfile >> nHidden; // retrieve number of hidden nodes
		getline( loadfile, lineString ); // " hidden nodes"
		setHidden( nHidden ); // set the architecture of this object

		getline( loadfile, lineString ); // "1 output node by definition"

		getline( loadfile, lineString ); // "Hidden layer:"
		loadfile >> hW; // retrieve hidden layer

		// Pick up the space and carriage return after the Matrix object
		getline( loadfile, lineString ); // THIS IS CRITICAL!

		getline( loadfile, lineString ); // "Output layer:"
		loadfile >> oW; // retrieve output layer

		weightsSetFlag = true; // set flag to indicate weights now set

		outputHeader( util::screen() ); // report to user
		util::screen() << "I've loaded the file." << endl;

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the output stream with object name and header
			ostringstream fileStream;
			fileStream << "I've loaded the file " << goodFile << ":" << endl;
			outputHeader( fileStream ); // output header to history file
			fileStream << endl;

			addHistory( fileStream ); // append to history file
		}

		success = true; // set flag to indicate file successfully loaded
	}

	loadfile.close(); // close input file

	return success; // return flag to indicate if file successfully loaded
}

// Inner training set algorithm called for automatic stepsize selection
//    calculates the gradient descent, and returns set error.
//
// ONE implementation for both architectures. It was ~185 lines written out
//    twice, and the executable difference was a single line -- the hidden
//    error term -- which is the SAME formula once its domain is written as a
//    range instead of assumed (see the note at that line). Everything the bias
//    architecture does decide happens elsewhere: forward() pins the bias slot,
//    setHidden() and setDataSet() size the structures, and this method reads
//    whatever sizes it was given.
double OneHiddenNet::innerTrainSet()
{
	// Easier on the eyes
	unsigned nTrain = theData.getNumTrain(), // examples in training set
		example; // example counter

	double setError = 0; // initialize the set error

	// For off-line training: zero'd accumulator containers for hidden
	//    and output weights--would be expensive, but it's *outside* the
	//    main loop, and so doesn't add much inefficiency to the on-line case
	Matrix< double > hWaccumulate( hW.rows(), hW.cols(), 0 );
	vector< double > oWaccumulate( oW.size(), 0 );

	// Reset average output error accumulator for automatic stepsize selection
	o_errAccumulate = 0.0;

	// Loop through all exemplars in the set
	for ( example = 0; example < nTrain; example++ )
	{
		// Begin by forward propagating the exemplar
		forward( Train, example );

		// Calculate error for single output and add to set error
		errorFunction E( y[ example ], o, x, errorType );
		setError += E.value();

		if ( E.boundsErr() ) // check for out of bounds error in log(o)
			boundsErrorFlag = true;

		// Regularization term for error, Manifest Methodology equation 2.1
		//    $\frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$ in
		//    $E_k({\bf y}) = e(t^k, {\bf a}^{(m)}_k) + \frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$
		if ( weightDecayFlag )
			setError += regularizer * ( hW.squared() + squared( oW ) );

		// Calculate the error term for the output unit
		// (Freeman & Skapura p. 102, #6)
		// $\delta^o_{pk} = (y_{pk} - o_{pk}){f^o_k}'(net^o_{pk})$ for mean squared error
		// Note sign is changed ( o - y ) instead of ( y - o ) to conform to Methodology
		// Methodology equation 2.8 ($t=y$, $a=o$)
		// $\delta^{(m)}_k = -({\bf t}_k - {\bf a}^{(m)})$ for mean squared error,
		// or equation 2.9
		// $\delta^{(m)}_k = -({\bf t}_k - {\bf a}^{(m)}) ./ [{\bf a}^{(m)} .* ({\bf 1} - {\bf a}^{(m)}]$
		//    for x-entropy error,
		// multiplied by the ${\bf Df}^{(j)}_k$ term in equation 2.7
		// Note that if the error is x-entropy, then dE/do = (y-o)/(o(1-o))
		//    and the o(1-o) in the denominator cancels d_sigmoidal
		//   ($={f^o_k}'$, $={\bf Df}^{(j)}_k$) = o(1-o)
		//    hence splitting this code into 2 lines with an if:
		o_err = o - y[ example ];
		if ( errorType == 0 ) // error is LMS
			o_err *= d_sigmoidal()( o );

		// If automatic stepsize selection, accumulate average error
		if ( batchEpochFlag && automaticStepSizeFlag )
			o_errAccumulate += o_err;

		// Calculate the error terms for the hidden units
		// (Freeman & Skapura p. 102, #7)
		// $\delta^h_{pj} = {f^h_j}'(net^h_{pj}) \sum_k \delta^o_{pk} w^o_{kj}$
		// Methodology equation 2.7
		// $\delta^{(j-1)}_k =  [{\bf W}^{(j)}]^T {\bf Df}^{(j)}_k \delta^{(j)}_k$
		// Note that using func which takes the output vector as the
		// last argument, and *= instead of *, is the most efficient way.
		// THE RANGE IS THE WHOLE HIDDEN LAYER, stated as a domain: elements
		//    0 .. nHidden - 1 are the hidden units. A biased model's hO carries
		//    one more element, the pinned bias slot, which is an input to the
		//    output unit and has no error term of its own; an unbiased model's
		//    hO ends at nHidden - 1, so the same range is its entire vector.
		//    One expression, two architectures -- not a flag.
		// The trailing *= oW relies on vector_ops' PREFIX RULE: h_err has
		//    nHidden elements and oW has one more in the biased model, and the
		//    surplus tail -- the output unit's own bias weight -- is ignored.
		( func( hO, d_sigmoidal(), h_err, 0, nHidden - 1 ) *= o_err ) *= oW;

		if ( !batchEpochFlag ) // classic on-line backpropagation
		{
			// Tests of gradient calculation were found to slow training
			//    by more than 50%, hence the if block
		// Weight decay for canonical backpropagation, applied ONCE PER WEIGHT
		//    UPDATE -- which is what the formula says:
		// $\vec w_{t+1} = (1-2\eta\lambda)\vec w_t - \eta \left. \frac{\partial E}{\partial w} \right|_{w_t}$
		//    On-line makes one update per exemplar, so it belongs beside the
		//    update below. It used to sit ABOVE the batch/on-line split and so
		//    ran once per EXEMPLAR in both modes: batch makes ONE update per
		//    epoch, so its effective per-epoch factor was (1-eta*decay)^N,
		//    exponential in the dataset size (D4; docs/refactor_audit.md section 9).
			if ( ( trainingType == 0 ) && !gradMaxFlag ) // where gradient doesn't need to be separated
			{
				if ( weightDecayFlag ) // once, for THIS exemplar's update
				{
					oW *= decayTerm;
					hW *= decayTerm;
				}

				// Update the output weights (Freeman & Skapura p. 102, #8)
				// $w^o_{kj}(t+1)=w^o_{kj}(t)+\eta\delta^o_{pk}i_{pj}$
				// Note that because the output error term is o-y as in Methodology,
				// it will be a subtraction, as in Methodology equation 2.11
				// ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}_k({\bf y}_t)$
				// where ${\bf g}_k = \delta^{(j)}_k{\bf Df}^{(j)}_k[{\bf f}^{(j-1)}_k]^T$
				// (Note that Freeman \& Skapura's $\delta^o_{pk}$ already contains ${\bf Df}^{(j)}_k$
				oW -= ( hO *= ( eta * o_err ) );
				// Update the hidden weights (Freeman & Skapura p. 102, #9)
				// $w^h_{ji}(t+1)=w^h_{ji}(t)+\eta\delta^h_{pj}x_i$
				// Also Methodology equation 2.11 as above (also a subtraction)
				hW -= ( hWup.outprod( h_err, I ) *= eta );
			}

			else // calculate the gradient as a separate structure
			{
				// Calculate the output and hidden gradients, Methodology equation 2.10
				// ${\bf g}_k = \delta^{(j)}_k{\bf Df}^{(j)}_k[{\bf f}^{(j-1)}_k]^T$
				// (Note that Freeman \& Skapura's $\delta$ already contains ${\bf Df}^{(j)}_k$)
				oG = ( hO *= o_err ); // *= for efficiency
				hG.outprod( h_err, I );

				// Weight decay, Manifest Methodology section 2.2.1
				// right hand term $(1/\sigma_w^2)[{\bf W} \;,\; {\bf b}]$
				// in ${\bf G}^{(j)}_k=\delta^{(j)}_k{\bf Df}^{(j)}_k[{\bf f}^{(j-1)}_k]^T+(1/\sigma_w^2)[{\bf W}\;,\;{\bf b}]$
				if ( weightDecayFlag )
				{
					oG += ( oW * decay );
					hG += ( hW * decay );
				}

				// Whatever additional algorithm is chosen
				engine( trainingType, ( iteration * nTrain ) + example );

				// Update the output and hidden weights
				// Methodology equation 2.11: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}_k({\bf y}_t)$
				oW -= ( oG * eta );
				hW -= ( hG * eta );
			}
		}

		else // off-line or batch/epoch learning
		{
			// Where gradient doesn't need to be separated
			if ( ( trainingType == 0 ) && !gradMaxFlag )
			{
				// Accumulate output weight update, see above note
				oWaccumulate += ( hO *= o_err );
				// Accumulate hidden weight update, see above note
				hWaccumulate += hWup.outprod( h_err, I );
			}

			else // calculate the gradient as a separate structure
			{
				// Calculate the output and hidden gradients, see above note
				oG = ( hO *= o_err );
				hG.outprod( h_err, I );

				// See above note in on-line block
				if ( weightDecayFlag )
				{
					oG += ( oW * decay );
					hG += ( hW * decay );
				}

				// Update the accumulators
				// Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
				oWaccumulate += oG;
				hWaccumulate += hG;
			}
		}
	} // end of loop for exemplars in training set

	if ( batchEpochFlag ) // off-line or batch/epoch learning
	{
		// Canonical backprop without separate gradient calculation
		if ( ( trainingType == 0 ) && !gradMaxFlag )
		{
			// Batch/epoch updates weights at the end, *now* multiply by eta
			// Methodology equation 2.13: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}({\bf y}_t)$
			// and $1/N$ in Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
			if ( weightDecayFlag ) // once, for THIS epoch's single update
			{
				oW *= decayTerm;
				hW *= decayTerm;
			}

			oW -= ( ( oWaccumulate *= eta ) / ( double ) nTrain ); // *=, /= for efficiency
			hW -= ( ( hWaccumulate *= eta ) / ( double ) nTrain );

		}

		else // routines where gradient is calculated separately
		{
			// Set the gradients to the accumulators so that pack() & unpack() work
			// $1/N$ in Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
			oG = oWaccumulate / ( double ) nTrain;
			hG = hWaccumulate / ( double ) nTrain;

			// Whatever additional algorithm is chosen
			engine( trainingType, iteration );

			// Update the output and hidden weights
			// Methodology equation 2.13: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}({\bf y}_t)$
			oW -= ( oG * eta );
			hW -= ( hG * eta );
		}
	}

	return setError / nTrain; // return the calculated set error
}

// Forward propagation from the exemplar already in I.
//
// PRECONDITION, and it is the whole bias architecture: the caller has read the
//    exemplar into I and, if this model has a bias slot in its hidden layer,
//    has pinned it. Both concrete forward() methods below do exactly that and
//    nothing else, so the statement that establishes a bias stays written in
//    the class that has one -- it is not a flag read here.
void OneHiddenNet::propagate()
{
	// Take the dot product of the hidden weights Matrix hW and the
	// transpose of a row of the input Matrix I, and apply the
	// sigmoidal function to the resulting vector
	// Elements 0 .. nHidden - 1 are the hidden units; a biased model's hO
	// carries the pinned bias slot after them, an unbiased model's does not
	// (Freeman & Skapura p. 102, #2 & #3)
	// 2: $net^h_{pj}=\sum_{i=1}^Nw^h_{ji}x_{pi}+\theta^h_j$, the bias term
	//    $\theta^h_j$ being the last column of hW in a biased model
	// 3: $i_{pj}=f_j^h(net^h_{pj})$
	func( hW.dotprod( I, hO, 0, nHidden - 1 ), sigmoidal(), hO, 0, nHidden - 1 );

	// Take the dotproduct of the hidden output vector and the output
	// weights vector, and apply the sigmoidal function to the result
	// (Freeman & Skapura p. 102, #4 & #5)
	// 4: $net^o_{pk}=\sum_{j=1}^Lw^o_{kj}i_{pj}+\theta^o_k$, the bias term
	//    $\theta^o_k$ being the last element of oW in a biased model
	// 5: $o_{pk}=f_k^o(net^o_{pk})$
	x = dotprod( hO, oW ); // note that x is inherited from Network
	o = sigmoidal()( x );
}

// Convert weight gradient structure to single vector stackG
void OneHiddenNet::pack()
{
	// Start by converting hidden gradients Matrix to stackG
	stackG = hG.toVector();

	// Then append output gradients vector to stackG
	std::copy( oG.begin(), oG.end(), back_inserter( stackG ) );
}
