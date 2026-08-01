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

// Convert weight gradient structure to single vector stackG
void OneHiddenNet::pack()
{
	// Start by converting hidden gradients Matrix to stackG
	stackG = hG.toVector();

	// Then append output gradients vector to stackG
	std::copy( oG.begin(), oG.end(), back_inserter( stackG ) );
}
