// neuron.cpp : The neuron 3.0 driver (successor to neUROn2++)
//

#include "stdafx.h" // For MSVC, must be first!

// Because, apparently, vector< vector< unsigned > > symbol names exceeds 255
//    characters in length, and MSCV will error with that (ugly!)
#ifdef _MSC_VER	// MSVC
#pragma warning (disable: 4786)
#endif

// #define GAURAV // For Gaurav's work on BackProp

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <memory>
#include <typeinfo>
#include "version.h"

#include "simpleprop.h" // concrete SimpleProp class
#include "bareprop.h"   // concrete BareProp class
#include "backprop.h"   // concrete BackProp class
#include "ldfa.h"       // concrete LDFA class
#include "qdfa.h"       // concrete QDFA class
#include "logistic.h"	// binary logistic regression
#include "dataset.h"    // DataSet for compound objects
#include "utility.h"    // utility methods
#include "regressnet.h" // network stepwise regression class
#include "gui.h"        // the embedded web GUI (neuron --gui)

using namespace std;

// Menu and submenu prototypes
unique_ptr< DataSet > specify_data(); // main menu: specify dataset method
	void data_characteristics( DataSet* ); // specify dataset: specify variate bounds & types
	bool randomize_raw( DataSet* ); // specify dataset: randomize raw dataset into training & test sets
	void specify_ROC( DataSet* ); // specify dataset: ROC reporting
	unique_ptr< Model > specify_model( DataSet* ); // main menu: specify model method
bool use_model( Model* ); // main menu: use model
	void stop_conditions( Model* ); // use model: specify stopping conditions
	void regress_menu( Model*, double ); // use model: stepwise regression menu
void dfa( DataSet* ); // main menu: discriminant function analysis

// Utility functions
void save_guesses( Model* ); // save guesses for a model with 1 output

// Model global log to file options
bool logLastOp = true, // initial log last operation to file flag
	logHistory = true; // initial log to history file flag

// Main method
int main( int argc, char* argv[] )
{
	// Parse command-line arguments
	bool guiFlag = false, // start the embedded web GUI instead of the menus
		browserFlag = true; // and open the browser on it

	for ( int a = 1; a < argc; a++ )
	{
		string arg( argv[ a ] );

		if ( arg == "--version" )
		{
			cout << NEURON_PACKAGE_STRING << endl;
			return 0;
		}
		else if ( arg == "--seed" && a + 1 < argc )
		{
			util::set_seed( ( unsigned ) atol( argv[ ++a ] ) );
			cout << "Random seed set to " << argv[ a ]
				<< " (runs with the same seed and inputs are reproducible)" << endl;
		}
		else if ( arg == "--gui" )
			guiFlag = true;
		else if ( arg == "--no-browser" )
			browserFlag = false;
		else
		{
			cerr << "Usage: neuron [--seed N] [--gui [--no-browser]] [--version]" << endl;
			return 1;
		}
	}

	if ( guiFlag ) // the web GUI replaces the menu loop entirely
		return run_gui( browserFlag );

	// Print welcome message
	cout << "Welcome to " << NEURON_PACKAGE_STRING << endl;

	// Dataset and Model owners for instantiated objects; smart pointers make
	//    replacement and scope exit exception-safe (no manual delete)
	unique_ptr< DataSet > dataPtr;
	unique_ptr< Model > modelPtr;

	// Set initial unsigned values
	unsigned choice = 0; // non-quit choice

	string reply; // for double-checking quit choice

	// Flags to indicate successful return from menu methods
	bool modelSpecifiedFlag = false,  // model not yet specified
		dataSpecifiedFlag = false,    // dataset not yet specified
		modelUsedFlag = false,        // model not yet used
		yesFlag;                      // for yes/no questions

	// Main menu loop
	while ( choice != 9 )
	{
		// Print main menu
		cout << endl << "( 1 ) Specify dataset" << endl;
		cout << "( 2 ) Specify model" << endl;
		cout << "( 3 ) Use model" << endl;
		cout << "( 4 ) Discriminant function analysis" << endl;
		cout << "( 9 ) Quit" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 9 );

		// Do choice
		if ( choice == 1 ) // specify dataset
		{
			// Get a new dataset object (replacing any previous one), test if valid
			if ( !( dataPtr = specify_data() ) )
				dataSpecifiedFlag = false;
			else
				dataSpecifiedFlag = true;

			if ( modelSpecifiedFlag ) // if the Model was previously specified
				modelPtr->setDataSet( *dataPtr ); // reload the dataset into it
		}
		else if ( choice == 2 ) // specify model
		{
			// Check to see if dataset has been specified, as dataset
			//    must be specified before model is specified
			if ( dataSpecifiedFlag )
			{
				// Get a new Model object (replacing any previous one), test if valid
				if ( !( modelPtr = specify_model( dataPtr.get() ) ) )
					modelSpecifiedFlag = false;
				else
					modelSpecifiedFlag = true;
				modelUsedFlag = false; // a new model will not have been used yet
			}
			else
				cout << "You must specify a dataset before a model." << endl;
		}
		else if ( choice == 3 ) // use the model
		{
			// Check to see if model and dataset have been specified
			if ( modelSpecifiedFlag && dataSpecifiedFlag )
				modelUsedFlag = use_model( modelPtr.get() );
			else
				cout << "You must specify a model and a dataset first." << endl;
		}
		else if ( choice == 4 ) // DFA
		{
			// Check to see that dataset specified
			if ( dataSpecifiedFlag )
				dfa( dataPtr.get() );
			else
				cout << "You must specify a dataset first." << endl;
		}
		else if ( choice == 9 ) // quit
		{
			// Double check quit choice
			yesFlag = util::askYesNo( "Are you sure you want to quit (yes/no)? " );
			if ( !yesFlag ) // if no
				choice = 0; // reset to non-quit choice
		}
		else
			cout << "I don't understand that choice." << endl;
	}

	// Print goodbye message
	cout << endl << "Thank you for using neuron 3.0" << endl;

	return 0;
}

// Main menu choice 1: Specify dataset submenu
unique_ptr< DataSet > specify_data() // returns a dataset if correctly specified
{
	auto dataPtr = make_unique< DataSet >(); // dataset factory

	// Set initial variable values
	unsigned choice = 0, // initial non-quit choice
		nI = dataPtr->getInput(), // number input nodes
		nO = dataPtr->getOutput(); // number output nodes when coded

	string reply; // for alpha replies
	ostringstream ostrReply; // for compound alpha replies

	bool yesFlag, // for yes/no questions
		scalesFlag = false; // to indicate that scales could be saved

	// Submenu loop
	while ( choice != 14 )
	{
		// Print submenu
		cout << endl << "( 1 ) Number of input nodes: " << nI << endl;
		cout << "( 2 ) Number of output nodes: " << nO << endl;
		cout << "( 3 ) Load raw dataset" << endl;
		cout << "( 4 ) Convert a raw dataset into a training set" << endl;
		cout << "( 5 ) Randomize a raw dataset into training and test sets" << endl;
		cout << "( 6 ) Load training set" << endl;
		cout << "( 7 ) Save training set" << endl;
		cout << "( 8 ) Save scaling factors" << endl;
		cout << "( 9 ) Load test set" << endl;
		cout << "( 10 ) Save test set" << endl;
		cout << "( 11 ) Log dataset operations to history file: ";
		if ( dataPtr->getHistory() )
			cout << "yes" << endl;
		else
			cout << "no" << endl;
		cout << "( 12 ) Specify variate bounds and types" << endl;
		if ( dataPtr->getDiscrete() && dataPtr->getOutput() == 1 )
			cout << "( 13 ) ROC area reporting" << endl;
		else
			cout << "( 13 ) N/A" << endl;
		cout << "( 14 ) Return to main menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 14 );

		// Do choice
		if ( choice == 1 ) // specify number input nodes
		{
			nI = util::askI( "How many input nodes? ", 1 );
			dataPtr->setInput( nI );
		}

		else if ( choice == 2 ) // specify number input nodes
		{
			nO = util::askI( "How many output nodes? ", 1 );
			dataPtr->setOutput( nO );
		}

		else if ( choice == 3 ) // load a raw dataset
		{
			// Get the filename for the raw data set
			cout << "What is the name of the raw data file? ";
			cin >> reply;

			dataPtr->loadRaw( reply ); // load the raw dataset from file
		}

		else if ( choice == 4 ) // convert a raw dataset into a training set
		{
			dataPtr->raw2train();
			scalesFlag = true; // scaling factors will be set by raw2train()

			// Query to save training set to file
			yesFlag = util::askYesNo( "Would you like to save the training set to a file? (yes/no)? " );
			if ( yesFlag )
			{
				cout << "What is the name of the file to save the training set? ";
				cin >> reply;
				dataPtr->saveTrain( reply ); // save the training set

				// To avoid great pain, if the user saves a training set converted from a raw dataset,
				//    they *must* save the scaling factors as well
				cout << "What is the name of the file to save the scaling factors? ";
				cin >> reply;
				dataPtr->saveScales( reply ); // save the scaling factors
			}
		}
		else if ( choice == 5 ) // randomize a raw dataset into training & test sets
		{
			if ( randomize_raw( dataPtr.get() ) ) // use submenu
				scalesFlag = true; // scaling factors will be set by randomize() or randomizeD()
		}

		else if ( choice == 6 ) // load a training set
		{
			// Get the filename for the training set
			cout << "What is the name of the training set data file? ";
			cin >> reply;

			// Load the file into the dataset
			dataPtr->loadTrain( reply );
		}

		else if ( choice == 7 ) // save the training set
		{
			if ( dataPtr->trainLoaded() ) // first insure training set is specified
			{
				cout << "What is the name of the file to save the training set? ";
				cin >> reply;
				dataPtr->saveTrain( reply ); // save the training set
			}
			else
				cout << "The training set must exist before it can be saved." << endl;
		}

		else if ( choice == 8 ) // save the scaling factors
		{
			if ( !scalesFlag ) // check to see if the scaling factors have been set
			{
				cout << "I'm sorry, but you must first convert a raw dataset to a training set" << endl;
				cout << "or randomize a dataset before saving scaling factors." << endl;
			}
			else // save the scaling factors
			{
				cout << "What is the name of the file to save the scaling factors? ";
				cin >> reply;
				dataPtr->saveScales( reply ); // save the scaling factors
			}
		}

		else if ( choice == 9 ) // load a test set
		{
			// Get the filename for the test set
			cout << "What is the name of the test set data file? ";
			cin >> reply;

			// Load the file into the dataset
			dataPtr->loadTest( reply );
		}

		else if ( choice == 10 ) // save the test set
		{
			if ( dataPtr->testLoaded() ) // first insure test set is specified
			{
				cout << "What is the name of the file to save the test set? ";
				cin >> reply;
				dataPtr->saveTest( reply ); // save the test set
			}
			else
				cout << "The test set must exist before it can be saved." << endl;
		}

		else if ( choice == 11 ) // set dataset history logging
		{
			yesFlag = util::askYesNo( "Would you like to log all dataset operations to a history file (yes/no)? " );
			if ( yesFlag )
				dataPtr->setHistory( true ); // set dataset history logging on
			else
				dataPtr->setHistory( false ); // set dataset history logging off
		}

		else if ( choice == 12 ) // specify variate bounds and types
			data_characteristics( dataPtr.get() );

		else if ( choice == 13 ) // specify number of ROC thresholds
			// Specify ROC reporting only if output is discrete and only 1 output
			if ( dataPtr->getDiscrete() && dataPtr->getOutput() == 1 )
				specify_ROC( dataPtr.get() );
	}

	if ( !dataPtr->trainLoaded() ) // if a training set wasn't loaded, it's not valid
		dataPtr.reset(); // so clear it (the 2.x code nulled the raw pointer here, leaking it)

	return dataPtr; // return the new dataset object (or empty on failure)
}

// Main menu choice 1: Specify dataset submenu choice: Randomize raw dataset
bool randomize_raw( DataSet* dataPtr )
{
	bool success = false, // flag to indicate if operation successful
		yesFlag; // for yes/no replies

	unsigned choice = 0, // menu choice
		n, d; // for unsigned replies

	double x; // for double replies

	string reply; // for alpha replies
	ostringstream ostrReply; // for compound alpha replies

	if ( !dataPtr->rawLoaded() ) // check that raw dataset exists
		cout << "Before a raw dataset is randomized, it must be loaded." << endl;

	else
	{
		// Print submenu
		cout << endl << "Specify number of exemplars to be placed in the test set:" << endl;
		cout << "( 1 ) As a whole number" << endl;
		cout << "( 2 ) As a fraction with a numerator and denominator" << endl;
		cout << "( 3 ) As a decimal fraction" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 3 );

		// Do choice
		if ( choice == 1 ) // specify test as whole number
		{
			// Get the number to be placed in the test set
			n = util::askI( "Number to be placed in test set = ", 0 );

			// Randomize the raw dataset
			success = dataPtr->randomize( n );
		}

		else if ( choice == 2 ) // specify test as a fraction
		{
			// Get the numerator of the fraction to be placed in test set
			n = util::askI( "Numerator of fraction = ", 0 );

			// Get the denominator of the fraction to be placed in the test set
			d = util::askI( "Denominator of fraction = ", 1 );

			// Randomize the raw dataset
			success = dataPtr->randomize( n, d );
		}

		else if ( choice == 3 ) // specify test as a decimal fraction
		{
			// Get the decimal fraction to be placed in the test set
			x = util::askD( "Fraction to be placed in test set = ", 0, 1 );

			// Randomize the raw dataset
			success = dataPtr->randomizeD( x );
		}

		// If successful, query to save training & test sets
		if ( success )
		{
			// Query to save training set to file
			yesFlag = util::askYesNo( "Would you like to save the training set to a file (yes/no)? " );
			if ( yesFlag )
			{
				cout << "What is the name of the file to save the training set? ";
				cin >> reply;
				dataPtr->saveTrain( reply ); // save the training set

				// To avoid great pain, if the user saves a training set from a randomized dataset,
				//    they *must* save the scaling factors as well
				cout << "What is the name of the file to save the scaling factors? ";
				cin >> reply;
				dataPtr->saveScales( reply ); // save the scaling factors
			}

			// Query to save test set to file
			yesFlag = util::askYesNo( "Would you like to save the test set to a file (yes/no)? " );
			if ( yesFlag )
			{
				cout << "What is the name of the file to save the test set? ";
				cin >> reply;
				dataPtr->saveTest( reply ); // save the test set
			}
		}
	}

	return success; // return flag to indicate if operation was successful
}

// Main menu choice 1: Specify dataset submenu choice: Specify variate bounds & types
void data_characteristics( DataSet* dataPtr )
{
	unsigned choice = 0; // initial non-quit choice

	double number; // for double replies

	string reply; // for alpha replies

	bool yesFlag; // for yes/no questions

	// Submenu loop
	while ( choice != 7 )
	{
		// Print submenu
		cout << endl << "( 1 ) Output is discrete (0 or 1): ";
		if ( dataPtr->getDiscrete() )
			cout << "yes" << endl;
		else
			cout << "no" << endl;
		cout << "( 2 ) Threshold for discrete output: ";
		if ( dataPtr->getDiscrete() )
			cout << dataPtr->getThreshold() << endl;
		else
			cout << "N/A" << endl;
		cout << "( 3 ) Lower limit of input variates: " << dataPtr->getInLower() << endl;
		cout << "( 4 ) Upper limit of input variates: " << dataPtr->getInUpper() << endl;
		cout << "( 5 ) Lower limit of output variates: ";
		if ( dataPtr->getDiscrete() )
			cout << "0" << endl;
		else
			cout << dataPtr->getOutLower() << endl;
		cout << "( 6 ) Upper limit of output variates: ";
		if ( dataPtr->getDiscrete() )
			cout << "1" << endl;
		else
			cout << dataPtr->getOutUpper() << endl;
		cout << "( 7 ) Return to previous menu" << endl; // returns to main menu

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 7 );

		// Do choice
		if ( choice == 1 ) // specify discrete or continuous output
		{
			yesFlag = util::askYesNo( "Would you like output to be discrete (held to 0,1) (yes/no)? " );
			dataPtr->setDiscrete( yesFlag );
		}

		else if ( choice == 2 ) // specify threshold for discrete output
		{
			if ( dataPtr->getDiscrete() ) // output must be discrete to specify threshold
			{
				cout << "To determine if a normalized output is discrete (0 to 1),"
					<< endl;
				double threshold = util::askD( "what will the threshold be? ", 0, 1 );
				dataPtr->setThreshold( threshold ); // set the threshold
			}
			else
				cout << "Output must be discrete to specify a threshold." << endl;
		}

		else if ( choice == 3 ) // specify lower limit of input variates
		{
			cout << "What will the lower limit of input variates be? ";
			cin >> number;
			dataPtr->setInLower( number ); // set the lower limit
		}

		else if ( choice == 4 ) // specify upper limit of input variates
		{
			cout << "What will the upper limit of input variates be? ";
			cin >> number;
			dataPtr->setInUpper( number ); // set the upper limit
		}

		else if ( choice == 5 ) // specify lower limit of output variates
		{
			if ( dataPtr->getDiscrete() ) // output must be nondiscrete
				cout << "To set a limit other than 0, turn discrete output off." << endl;
			else
			{
				cout << "What will the lower limit of output variates be? ";
				cin >> number;
				dataPtr->setOutLower( number ); // set the lower limit
			}
		}

		else if ( choice == 6 ) // specify upper limit of output variates
		{
			if ( dataPtr->getDiscrete() ) // output must be nondiscrete
				cout << "To set a limit other than 1, turn discrete output off." << endl;
			else
			{
				cout << "What will the upper limit of output variates be? ";
				cin >> number;
				dataPtr->setOutUpper( number ); // set the upper limit
			}
		}
	}
}

// Main menu choice 1: Specify dataset submenu choice: Specify ROC reporting
void specify_ROC( DataSet* dataPtr )
{
	cout << endl << "WARNING: DO NOT CHANGE THESE UNLESS YOU KNOW WHAT YOU'RE DOING" << endl;

	unsigned choice = 0, // initial non-quit choice
		quitChoice = 3, // final menu (quit) choice
		n; // for unsigned replies

	bool yesFlag; // for yes/no questions

	// For right justified, columnated printing
	#define COUT12 cout << setw( 12 ) << setfill( ' ' ) << setiosflags( ios::right )
	#define COUT15 cout << setw( 15 ) << setfill( ' ' ) << setiosflags( ios::right )

	// Submenu loop
	while ( choice != quitChoice )
	{
		// Print submenu
		cout << endl << "Remember: If you reload a training/test set, these will be reset" << endl
			<<  "                                                 Training Set       Test Set" << endl;
		// Choose if either or both of trapezoidal or statistical method is used
		cout << "( 1 ) Statistical or trapezoidal method:         ";
		if ( dataPtr->trainLoaded() )
		{
			if ( dataPtr->getTrainTwoSet().getROCReportFlag() )
				COUT12 << "both";
			else
				COUT12 << "either";
		}
		else
			COUT12 << "not loaded";
		if ( dataPtr->testLoaded() )
		{
			if ( dataPtr->getTestTwoSet().getROCReportFlag() )
				COUT15 << "both" << endl;
			else
				COUT15 << "either" << endl;
		}
		else
			COUT15 << "not loaded" << endl;
		// Choose minimum number of data points for statistical calculation
		cout << "( 2 ) Minimum data for statistical calculation:  ";
		if ( dataPtr->trainLoaded() )
			COUT12 << dataPtr->getTrainTwoSet().getROCthresh();
		else
			COUT12 << "not loaded";
		if ( dataPtr->testLoaded() )
			COUT15 << dataPtr->getTestTwoSet().getROCthresh() << endl;
		else
			COUT15 << "not loaded" << endl;
		// Quit choice
		cout << "( " << quitChoice << " ) Return to main menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, quitChoice );

		if ( choice != quitChoice )
		{
			// Make sure that training/test set is loaded
			unsigned tSet = util::askI( "For the training (1), the test (2) set, or both (3)? ", 1, 3 );
			if ( ( ( tSet == 1 ) || ( tSet == 3 ) ) && !dataPtr->trainLoaded() )
				cout << "Training set must first be loaded" << endl;
			else if ( ( ( tSet == 2 ) || ( tSet == 3 ) ) && !dataPtr->testLoaded() )
				cout << "Test set must first be loaded" << endl;
			else
			{
				// Do choice
				if ( choice == 1 ) // specify if either or both of trap/stat method used
				{
					n = util::askI( "Use both (1) or either (2) statistical/trapezoidal method(s)? ", 1, 2 );
					if ( ( tSet == 1 ) || ( tSet == 3 ) )
					{
						if ( n == 1 ) // both
							dataPtr->getTrainTwoSet().setROCReportFlag( true );
						else // either
							dataPtr->getTrainTwoSet().setROCReportFlag( false );
					}
					if ( ( tSet == 2 ) || ( tSet == 3 ) )
					{
						if ( n == 1 ) // both
							dataPtr->getTestTwoSet().setROCReportFlag( true );
						else // either
							dataPtr->getTestTwoSet().setROCReportFlag( false );
					}
				}
				if ( choice == 2 ) // minimum data for statistical calculation
				{
					n = util::askI( "What are the minimum number of data for statistical ROC calculation? ", 2 );
					if ( ( tSet == 1 ) || ( tSet == 3 ) )
						dataPtr->getTrainTwoSet().setROCthresh( n );
					if ( ( tSet == 2 ) || ( tSet == 3 ) )
						dataPtr->getTestTwoSet().setROCthresh( n );
				}
			} // end of do choice
		} // end of if ( choice != quitChoice )
	} // end of while ( choice != quitChoice )

	#undef COUT12
	#undef COUT15
}

// Main menu choice 2: Specify model submenu
unique_ptr< Model > specify_model( DataSet* dataPtr ) // returns a Model if correctly specified
{
	unique_ptr< Model > modelPtr; // model factory

	// Set initial variable values
	unsigned choice = 0, // initial non-quit choice
		// If output discrete, initial choice output error function is X-entropy, otherwise LMS
		errorChoice = ( dataPtr->getDiscrete() ? 2 : 1 ),
		nLayers, // number of hidden node layers
		i; // the usual iterator

	vector< unsigned > nLayer; // number of nodes on each layer
	nLayer.push_back( 1 ); // initial number of layers & nodes

	bool biases = true, // initial biases
		yesFlag; // for yes/no replies

	string reply, // for alpha replies
		lineString; // for lines from a file

	// Submenu loop
	while ( choice != 8 )
	{
		// Print submenu
		cout << endl << "( 1 ) Bias nodes: ";
		if ( biases )
			cout << "yes" << endl;
		else
			cout << "no" << endl;
		cout << "( 2 ) Number of hidden layers: " << nLayer.size() << endl;
			for ( i = 0; i < nLayer.size(); i++ )
				cout << "         Layer " << i + 1 << ": " << nLayer[ i ]
				   << " node(s)" << endl;
		cout << "( 3 ) Output error function: ";
		if ( errorChoice == 1 )
			cout << "LMS" << endl;
		else if ( errorChoice == 2 )
			cout << "X-entropy" << endl;
		cout << "( 4 ) Load network from a file" << endl;
		cout << "( 5 ) Binary logistic regression" << endl;
		cout << "( 6 ) Log last operation to file: ";
		if ( logLastOp )
			cout << "yes" << endl;
		else
			cout << "no" << endl;
		cout << "( 7 ) Log to history file: ";
		if ( logHistory )
			cout << "yes" << endl;
		else
			cout << "no" << endl;
		cout << "( 8 ) Return to main menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 8 );

		// Do choice
		if ( choice == 1 ) // specify biases
			biases = util::askYesNo( "Would you like bias nodes (yes/no)? " );

		else if ( choice == 2 ) // specify hidden layers
		{
			nLayers = util::askI( "How many hidden layers? ", 1 );
			nLayer.resize( nLayers ); // set size of hidden layer vector
			for ( i = 0; i < nLayers; i++ )
			{
				cout << "For layer " << i + 1 << ", ";
				nLayer[ i ] = util::askI( "how many hidden nodes? ", 1 );
			}
		}
		else if ( choice == 3 ) // choose output error function
		{
			// Print submenu
			cout << endl << "Choose output error:" << endl
				<< "( 1 ) Least mean squared" << endl
				<< "( 2 ) X-entropy" << endl;

			unsigned oldChoice = errorChoice; // remember the original choice

			// Query user for choice
			errorChoice = util::askI( "Your choice: ", 1, 2 );

			// Insure appropriate type of output
			if ( errorChoice == 2 && !dataPtr->getDiscrete() )
			{
				cout << "I'm sorry, but to select that, output must be discrete." << endl;
				errorChoice = oldChoice; // reset to original choice
			}
		}

		else if ( choice == 4 ) // load network from file
		{
			// Flag to indicate type of object identified in file is valid
			bool validModel = true;

			// Query the user for the file name
			cout << "What is the name of the file that contains the network? ";
			cin >> reply;

			// Open the file as read-only, and associate it with the 'label' loadfile
			ifstream loadfile( ( util::getGoodFile( reply ) ).c_str(), ios::in );

			// Get the type of object from the 1st line of file
			getline( loadfile, lineString );

			util::chopEndl( lineString ); // remove <cr> from end of string if exists

			// Make a new Model to load the file into
			if ( lineString == "Binary logistic" ) // it's a Logistic object
				modelPtr = make_unique< Logistic >();
			else if ( lineString == "SimpleProp" ) // it's a SimpleProp object
				modelPtr = make_unique< SimpleProp >();
			else if ( lineString == "BareProp" ) // it's a BareProp object
				modelPtr = make_unique< BareProp >();
			else if ( lineString == "BackProp" ) // it's a BackProp object
			{
				modelPtr = make_unique< BackProp >();

				// Determine if network has biases from next line in file
				bool biasFlag;
				loadfile >> biasFlag;

				dynamic_cast< Network* >( modelPtr.get() )->setBias( biasFlag );
			}
			else
				validModel = false; // no valid object type was found

			loadfile.close(); // and close the input file

			if ( validModel ) // if a valid object type was found
			{
				// Load the dataset: this must happen BEFORE architecture is specified
				modelPtr->setDataSet( *dataPtr );

				// Specify logging flags
				modelPtr->setLastop( logLastOp );
				modelPtr->setHistory( logHistory );

				// Set output error function in Model object
				if ( lineString != "Binary logistic" ) // X-entropy by definition if binary logistic
				{
					if ( errorChoice == 1 ) // LMS error
						modelPtr->setLMSerror();
					else if ( errorChoice == 2 ) // X-entropy error
						modelPtr->setXEerror();
				}

				// Use the Network method to load it from file
				if ( dynamic_cast< Network* >( modelPtr.get() )->load( reply ) )
					// Return directly to main menu to avoid redefining model
					return modelPtr; // return a pointer to the Model
			}
			else // no valid object type was found
				cout << "Sorry, that file contains an unknown type." << endl;
		}

		else if ( choice == 5 ) // binary logistic regression
		{
			// Check to insure only 1 discrete output
			if ( ( dataPtr->getOutput() != 1 ) || !dataPtr->getDiscrete() )
				cout << "For logistic regression, there can be only 1 output, and it must be discrete."
					<< endl;

			else // check's OK, so make object
			{
				modelPtr = make_unique< Logistic >();

				// And load its dataset: this must happen BEFORE architecture is specified
				modelPtr->setDataSet( *dataPtr );

				// Specify logging flags
				modelPtr->setLastop( logLastOp );
				modelPtr->setHistory( logHistory );

				// Return directly to main menu to avoid redefining model
				return modelPtr; // return a pointer to the Model
			}
		}

		else if ( choice == 6 ) // log last operation to file
		{
			yesFlag = util::askYesNo( "Would you like to log the last operation to a file (yes/no)? " );
			if ( yesFlag )
				logLastOp = true; // set last operation logging on
			else
				logLastOp = false; // set last operation logging off
		}

		else if ( choice == 7 ) // log to history file
		{
			yesFlag = util::askYesNo( "Would you like to log to a history file (yes/no)? " );
			if ( yesFlag )
				logHistory = true; // set history logging on
			else
				logHistory = false; // set history logging off
		}
	}

	// The Model factory
#ifdef GAURAV // general backpropagation network
	modelPtr = make_unique< BackProp >();
	dynamic_cast< Network* >( modelPtr.get() )->setBias( biases );
#else // Determine the type of Model object
	if ( dataPtr->getOutput() == 1 && nLayer.size() == 1 )
	{
		if ( biases )
			modelPtr = make_unique< SimpleProp >(); // biases makes it SimpleProp
		else
			modelPtr = make_unique< BareProp >(); // no biases makes it BareProp
	}
	else
	{
		modelPtr = make_unique< BackProp >(); // general backpropagation network
		dynamic_cast< Network* >( modelPtr.get() )->setBias( biases );
	}
#endif

	// Specify logging flags
	modelPtr->setLastop( logLastOp );
	modelPtr->setHistory( logHistory );

	// And load its dataset: this must happen BEFORE architecture is specified
	modelPtr->setDataSet( *dataPtr );

	// Set output error function in Model object
	if ( errorChoice == 1 ) // LMS error
		modelPtr->setLMSerror();
	else if ( errorChoice == 2 ) // X-entropy error
		modelPtr->setXEerror();

	// Set the number of hidden nodes to specify the network architecture
#ifdef GAURAV // general backpropagation network
	dynamic_cast< BackProp* >( modelPtr.get() )->setHidden( nLayer );
#else
	if ( ( dataPtr->getOutput() == 1 ) && ( nLayer.size() == 1 ) )
	{
		if ( biases )
			dynamic_cast< SimpleProp* >( modelPtr.get() )->setHidden( nLayer[ 0 ] );
		else
			dynamic_cast< BareProp* >( modelPtr.get() )->setHidden( nLayer[ 0 ] );
	}
	else
		dynamic_cast< BackProp* >( modelPtr.get() )->setHidden( nLayer );
#endif

	return modelPtr; // return pointer to new Model object
}

// Main menu choice 3: Use model submenu
bool use_model( Model* modelPtr ) // returns true if dataset modeled
{
	// Set initial variable values
	unsigned choice = 0, // initial non-quit choice
		// Initial linear print count
		linPrintCount = dynamic_cast< Iterative* >( modelPtr )->getPrintCount();

	double eta, // learning rate
		decay = dynamic_cast< Network* >( modelPtr )->getDecay(), // lambda for weight decay
		modelError = -1; // final error of trained model

	bool modeled = false, // model used flag
		logPrint = dynamic_cast< Iterative* >( modelPtr )->getLogPrint(), // flag for log printing
		epochFlag = dynamic_cast< Network* >( modelPtr )->getBatchEpoch(), // flag for batch/epoch
		decayFlag = dynamic_cast< Network* >( modelPtr )->getWeightDecay(), // flag for weight decay
		autoStepFlag = dynamic_cast< Network* >( modelPtr )->getAutoStepSize(), // flag for automatic stepsize
		yesFlag; // for yes/no questions

	// If a network has been already loaded, train 1 iteration with eta 0 to get current error
	if ( dynamic_cast< Network* >( modelPtr )->getWeightsSet() )
	{
		// Get initial number iterations and eta
		unsigned currMaxIter = dynamic_cast< Iterative* >( modelPtr )->getMaxIterations();
		double currEta = dynamic_cast< Network* >( modelPtr )->getEta();

		// Set number iterations to 1, eta to 0
		dynamic_cast< Iterative* >( modelPtr )->setMaxIterations( 1 );
		dynamic_cast< Network* >( modelPtr )->setEta( 0 );

		// Do a canonical backprop iteration with eta 0
		dynamic_cast< Network* >( modelPtr )->setTrainingType( 0 );
		modelError = modelPtr->train(); // train the model, record error
		modeled = true; // the model was trained

		// Replace initial number iterations and eta
		dynamic_cast< Iterative* >( modelPtr )->setMaxIterations( currMaxIter );
		dynamic_cast< Network* >( modelPtr )->setEta( currEta );
	}

	// Set the initial learning rate
	if ( epochFlag )
		eta = 1.0;
	else
		eta = 1.0 / ( double ) modelPtr->getDataSet().getNumTrain();
	dynamic_cast< Network* >( modelPtr )->setEta( eta );

	string reply; // for alpha replies
	ostringstream ostrReply; // for compound alpha replies

	// Submenu loop
	while ( choice != 10 )
	{
		// Print submenu
		cout << endl << "( 1 ) Randomize weights" << endl;
		cout << "( 2 ) Set learning rate: ";
		if ( autoStepFlag )
			cout << "automatic" << endl;
		else
			cout << eta << endl;
		cout << "( 3 ) Set stopping conditions" << endl;
		cout << "( 4 ) Set batch/epoch (currently ";
		if ( epochFlag )
			cout << "on)" << endl;
		else
			cout << "off)" << endl;
		cout << "( 5 ) Set weight decay (";
		if ( decayFlag )
			cout << "lambda = " << decay << ")" << endl;
		else
			cout << "currently off)" << endl;
		cout << "( 6 ) Print counter: ";
		if ( logPrint )
			cout << "logarithmic" << endl;
		else
			cout << linPrintCount << endl;
		cout << "( 7 ) Train model" << endl;
		cout << "( 8 ) Save the network to a file" << endl;
		cout << "( 9 ) Stepwise Regression" << endl;
		cout << "( 10 ) Return to main menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 10 );

		// Do choice
		if ( choice == 1 ) // randomize weights
			dynamic_cast< Network* >( modelPtr )->randomize();

		else if ( choice == 2 ) // get & set the learning rate
		{
			// Query for and set automatic stepsize selection
			autoStepFlag = util::askYesNo( "Would you like the learning rate to be automatic? (yes/no)? " );
			dynamic_cast< Network* >( modelPtr )->setAutoStepSize( autoStepFlag );

			if ( autoStepFlag )
			{
				if ( !epochFlag ) // batch epoch must be chosen for auto stepsize
				{
					cout << "Batch epoch is required for an automatic learning rate, and will be set on." << endl;
					epochFlag = true;
					dynamic_cast< Network* >( modelPtr )->setBatchEpoch( epochFlag );

				}
			}

			else // no auto stepsize, so suggest learning rates
			{
				// Suggest learning rates based on on- or off-line training
				if ( epochFlag ) // batch/epoch is on
				{
					yesFlag = util::askYesNo( "Batch/epoch is on.  Set learning rate to 1 (yes/no)? " );
					if ( yesFlag )
						eta = 1.0;
					else
						eta = util::askD( "What is the learning rate? ", 0, 1 );

					dynamic_cast< Network* >( modelPtr )->setEta( eta );
				}
				else // batch/epoch is off
				{
					eta = 1.0 / ( double ) modelPtr->getDataSet().getNumTrain(); // suggest eta = 1/N

					ostrReply.str( "" ); // clear the compound query ostringstream
					ostrReply << "Batch/epoch is off.  Set learning rate to 1/N = "
						<< eta << " (yes/no)? ";
					yesFlag = util::askYesNo( ostrReply.str() );

					if ( !yesFlag ) // if user doesn't want suggestion
						eta = util::askD( "What is the learning rate? ", 0, 1 );

					dynamic_cast< Network* >( modelPtr )->setEta( eta ); // set the learning rate
				}
			}
		}
		else if ( choice == 3 ) // set stopping conditions
			stop_conditions( modelPtr );

		else if ( choice == 4 ) // batch/epoch
		{
			// If automatic stepsize is selected, batch/epoch must be selected
			if ( autoStepFlag )
			{
				cout << "Automatic learning rate is on, batch/epoch must be on." << endl;
				epochFlag = true;
				dynamic_cast< Network* >( modelPtr )->setBatchEpoch( epochFlag );
			}

			// If it's a binary logistic network, batch/epoch must be selected
			else if ( typeid( *modelPtr ) == typeid( Logistic ) )
			{
				cout << "For logistic regression, batch/epoch must be on." << endl;
				epochFlag = true;
				dynamic_cast< Network* >( modelPtr )->setBatchEpoch( epochFlag );
			}

			else // query and set batch/epoch
			{
				epochFlag = util::askYesNo( "Would you like batch/epoch (offline) training (yes/no)? " );
				dynamic_cast< Network* >( modelPtr )->setBatchEpoch( epochFlag );

				// Suggest learning rates
				if ( epochFlag && ( eta != 1 ) ) // 1 for batch/epoch
				{
					yesFlag = util::askYesNo( "Would you like to set the learning rate to 1 (yes/no)? " );
					if ( yesFlag )
					{
						eta = 1;
						dynamic_cast< Network* >( modelPtr )->setEta( 1 );
					}
					else
						cout << "O.K., I'll keep it at " << eta << "." << endl;
				}
				else if ( !epochFlag ) // 1/N for online learning
				{
					double newEta = 1.0 / ( double ) modelPtr->getDataSet().getNumTrain();

					if ( newEta != eta )
					{
						ostrReply.str( "" ); // clear the compound query ostringstream
						ostrReply << "Would you like to set the learning rate to 1/N = "
							<< newEta << " (yes/no)? ";
						yesFlag = util::askYesNo( ostrReply.str() );

						if ( yesFlag )
						{
							eta = newEta;
							dynamic_cast< Network* >( modelPtr )->setEta( newEta ); // set the learning rate
						}
						else
							cout << "O.K., I'll keep it at " << eta << "." << endl;
					}
				}
			}
		}
		else if ( choice == 5 ) // weight decay
		{
			// Choose way to calculate lambda
			cout << endl << "Don't change this unless you've read the manifest" << endl;
			cout << "and you know what you're doing!" << endl;
			cout << "( 1 ) Don't change anything" << endl;
			cout << "( 2 ) lambda = 1 / ( 2 * sigma^2 )" << endl;
			cout << "( 3 ) lambda = 1 / ( 2 * N * sigma^2 )" << endl;
			cout << "( 4 ) No weight decay" << endl;
			choice = util::askI( "Menu choice: ", 1, 4 );

			if ( ( choice == 2 ) || ( choice == 3 ) ) // weight decay
			{
				unsigned power = util::askI( "Choose a power of 10 (10^1 to 10^12) for sigma: ", 1, 12 );

				if ( choice == 2 ) // lambda = 1 / ( 2 * sigma^2 )
					decay = 1.0 / pow( pow( double( 10 ), double( power ) ), double( 2 ) ); // note that decay = 2 * lambda
				else // lambda = 1 / ( 2 * N * sigma^2 )
				{
					double N = ( double ) modelPtr->getDataSet().getNumTrain();
					decay = 1.0 / ( N * pow( pow( double( 10 ), double( power ) ), double( 2 ) ) ); // note that decay = 2 * lambda
				}

				// Set the weight decay flag in the Network object
				decayFlag = true;
				dynamic_cast< Network* >( modelPtr )->setWeightDecay( true );

				// Set the decay term lambda in the Network object
				dynamic_cast< Network* >( modelPtr )->setDecay( decay );
			}
			else if ( choice == 4 )
			{
				// Turn off the weight decay flag in the Network object
				decayFlag = false;
				dynamic_cast< Network* >( modelPtr )->setWeightDecay( false );

				// Zero the decay term lambda in the Network object
				decay = 0;
				dynamic_cast< Network* >( modelPtr )->setDecay( 0 );
			}
		}
		else if ( choice == 6 ) // get & set type and value of print counter
		{
			// Query if user wants a logarithmic print counter
			logPrint = util::askYesNo( "Would you like a logarithmic print counter (yes/no)? " );
			dynamic_cast< Iterative* >( modelPtr )->setLogPrint( logPrint );

			if ( !logPrint ) // linear print counter
			{
				// Query for the linear print counter
				linPrintCount = util::askI( "What will the linear print count be? ", 1 );
				// Set the linear print count
				dynamic_cast< Iterative* >( modelPtr )->setPrintCount( linPrintCount );
			}
		}
		else if ( choice == 7 ) // train the model
		{
			// Check to see if weights are set before model is used
			if ( dynamic_cast< Network* >( modelPtr )->getWeightsSet() )
			{
				// Choose training type
				cout << "Choose training algorithm:" << endl;
				cout << "( 1 ) Canonical backpropagation (gradient descent)" << endl;
				cout << "( 2 ) Conjugate gradient descent" << endl;
				cout << "( 3 ) Shanno's algorithm" << endl;
				choice = util::askI( "Menu choice: ", 1, 3 );

				// Warn user if training type may not be appropriate
				yesFlag = true;
				if ( !epochFlag && ( ( choice == 2 ) || ( choice == 3 ) ) )
				{
					cout << "WARNING: Batch/epoch training is recommended for Shanno's or CGD" << endl;
					yesFlag = util::askYesNo( "Are you sure you want to proceed (yes/no)? " );
				}

				if ( yesFlag )
				{
					// Set the training type (it's indexed from 0, *not* from 1)
					dynamic_cast< Network* >( modelPtr )->setTrainingType( choice - 1 );

					modelError = modelPtr->train(); // train the model, record error
					modeled = true; // the model was trained

					// Query if user wants to save model
					yesFlag = util::askYesNo( "Would you like to save this model (yes/no)? " );
					if ( yesFlag )
					{
						cout << "What is the name of the file to save the network? ";
						cin >> reply;
						dynamic_cast< Network* >( modelPtr )->save( reply ); // save the model

						// Save model guesses if desired
						save_guesses( modelPtr );
					}
				}
			}
			else
				cout << "Weights must be set before network can be used." << endl;
		}
		else if ( choice == 8 ) // save the network
		{
			// Check to see if weights are set before network is saved
			if ( dynamic_cast< Network* >( modelPtr )->getWeightsSet() )
			{
				// Check to see if the network has been trained
				if ( modelError == -1 )
					cout << "The network must be trained before it can be saved." << endl;
				else
				{
					cout << "What is the name of the file to save the network? ";
					cin >> reply;
					dynamic_cast< Network* >( modelPtr )->save( reply ); // save the model

					// Save model guesses if desired
					save_guesses( modelPtr );
				}
			}
			else
				cout << "Weights must be set before network can be saved." << endl;
		}
		else if ( choice == 9 ) // stepwise regression
		{
			// Make sure weights set
			if ( !dynamic_cast< Network* >( modelPtr )->getWeightsSet() )
				cout << "Network weights must be set before stepwise regression." << endl;
			else // pass last error
			{
				if ( modelError == -1 ) // the final XE-error needs to be set
					modelError = util::askD( "What was the final cross-entropy error? ", 0, 1 );

				regress_menu( modelPtr, modelError );
			}
		}
	}

	return modeled; // return flag if model used
}

// Main menu choice 3: Use model submenu choice: Set stop conditions
void stop_conditions( Model* modelPtr )
{
	// Set initial variable values
	unsigned choice = 0, // initial non-quit choice
		// Initial maximum iterations
		maxIter = dynamic_cast< Iterative* >( modelPtr )->getMaxIterations(),
		// Initial error window width
		errWindow = dynamic_cast< Iterative* >( modelPtr )->getWindow();

	// Flag if min error used to stop training
	bool lowErrFlag = dynamic_cast< Iterative* >( modelPtr )->getMinStop(),
		// Flag if change in error over 1 iteration used to stop training
		changeFlag = dynamic_cast< Iterative* >( modelPtr )->getChangeStop(),
		// Flag if error increases over a window used to stop training
		errWinFlag = dynamic_cast< Iterative* >( modelPtr )->getWindowStop(),
		// Flag if maximum absolute gradient decreases below a limit
		gradMaxFlag = dynamic_cast< Iterative* >( modelPtr )->getGradStop();

	// Min error value
	double lowErr = dynamic_cast< Iterative* >( modelPtr )->getMinError(),
		// Change over 1 iteration value
		change = dynamic_cast< Iterative* >( modelPtr )->getChange(),
		// Maximum absolute gradient limit
		gradMaxLimit = dynamic_cast< Iterative* >( modelPtr )->getGradMaxLimit();

	string reply; // for alpha replies

	// Submenu loop
	while ( choice != 6 )
	{
		// Print submenu
		cout << endl << "( 1 ) Set maximum number of iterations: " << maxIter << endl;
		cout << "( 2 ) Lower error limit: ";
		if ( lowErrFlag )
			cout << lowErr << endl;
		else
			cout << "no" << endl;
		cout << "( 3 ) Limit of change in error over 1 iteration: ";
		if ( changeFlag )
			cout << change << endl;
		else
			cout << "no" << endl;
		cout << "( 4 ) Stop if increase over error window width: ";
		if ( errWinFlag )
			cout << errWindow << endl;
		else
			cout << "no" << endl;
		cout << "( 5 ) Maximum absolute gradient limit: ";
		if ( gradMaxFlag )
			cout << gradMaxLimit << endl;
		else
			cout << "no" << endl;
		cout << "( 6 ) Return to previous menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 6 );

		// Do choice
		if ( choice == 1 ) // get & set maximum number of iterations
		{
			maxIter = util::askI( "What will the maximum number of iterations be? ", 1 );
			// Set the maximum number of iterations
			dynamic_cast< Iterative* >( modelPtr )->setMaxIterations( maxIter );
		}
		else if ( choice == 2 ) // set minimum error condition
		{
			// Query if user wants a minimum error
			lowErrFlag = util::askYesNo( "Would you like to stop at a minimum error (yes/no)? " );
			dynamic_cast< Iterative* >( modelPtr )->setMinStop( lowErrFlag );

			if ( lowErrFlag )
			{
				// Query for the minimum error value, and set it
				lowErr = util::askD( "What will the minimum error be? ", 0, 1 );
				dynamic_cast< Iterative* >( modelPtr )->setMinError( lowErr );
			}
		}
		else if ( choice == 3 ) // set change in error condition
		{
			// Query if user wants to stop if error changes over 1 iteration
			changeFlag = util::askYesNo( "Stop if error changes over 1 iteration (yes/no)? " );
			dynamic_cast< Iterative* >( modelPtr )->setChangeStop( changeFlag );

			if ( changeFlag )
			{
				// Query for the change in error value, and set it
				change = util::askD( "What will the limit of change in error be? ", 0, 1 );
				dynamic_cast< Iterative* >( modelPtr )->setChange( change );
			}
		}
		else if ( choice == 4 ) // set error window condition
		{
			// Query if user wants an error window width
			errWinFlag = util::askYesNo( "Would you like to stop if error increases over a window (yes/no)? " );
			dynamic_cast< Iterative* >( modelPtr )->setWindowStop( errWinFlag );

			if ( errWinFlag )
			{
				// Query for the window width, and set it
				errWindow = util::askI( "What will the window width be? ", 1 );
				dynamic_cast< Iterative* >( modelPtr )->setWindow( errWindow );
			}
		}
		else if ( choice == 5 ) // set maximum absolute gradient
		{
			// Query if user wants a maximum absolute gradient
			gradMaxFlag = util::askYesNo( "Would you like to stop at a maximum absolute gradient limit (yes/no)? " );
			dynamic_cast< Iterative* >( modelPtr )->setGradStop( gradMaxFlag );

			if ( gradMaxFlag )
			{
				// Query for the maximum absolute gradient, and set it
				gradMaxLimit = util::askD( "What will the maximum absolute gradient limit be? ", 0 );
				dynamic_cast< Iterative* >( modelPtr )->setGradMaxLimit( gradMaxLimit );
			}
		}
	}
}

// Main menu choice 3: Use model submenu choice: stepwise regression menu, pass pointer
//    to model & its final error
void regress_menu( Model* modelPtr, double lastError )
{
	RegressNet regressionObj; // object to stepwise regress a Network

	// Make Network pointer to Model object & set in stepwise regression object
	Network* netPtr = dynamic_cast< Network* >( modelPtr );
	regressionObj.setNetwork( netPtr, lastError );

	unsigned choice = 0, // menu choices
		sub_choice = 0;

	vector< vector< unsigned > > variable_defs; // variable definitions structure

	string variable_string, // for entry of variable definitions
		reply; // for alpha replies

	bool variables_specified = false; // successful entry of variable definitions

	// Submenu loop
	while ( choice != 4 )
	{
		// Print submenu
		cout << endl << "( 1 ) Specify variables" << endl;
		cout << "( 2 ) Stepwise reverse regression" << endl;
		cout << "( 3 ) Stepwise forward regression" << endl;
		cout << "( 4 ) Return to previous menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 4 );

		// Do choice
		if ( choice == 1 ) // Specify variables
		{
			variables_specified = false; // reset success flag

			cout << endl << "( 1 ) From the command line" << endl;
			cout << "( 2 ) From a file" << endl;
			sub_choice = util::askI( "Choose: ", 1, 2 );
			if ( sub_choice == 1 ) // from command line
			{
				// Get input string, & use utility.h variable_parse method to
				//    construct variable definition structure
				cout << endl << "Variables are separated by semicolons (;)," << endl
					<< "input nodes by commas (,), and hyphens can be used" << endl
					<< "for multiple input nodes.  For example: 0-4;5,6;7;8" << endl
					<< endl << "Enter definition of variables: ";
				cin >> variable_string;
				try { variable_defs = util::variable_parse( variable_string ); }
				catch ( util::utilErr& e ) { cout << e.what() << endl; }
			}
			else // from a file
			{
				// Get the filename with the variables specification string & parse
				cout << endl << "What is the name of the file? ";
				cin >> reply;
				ifstream iFile( ( util::getGoodFile( reply ) ).c_str(), ios::in );
				try { variable_defs = util::variable_parse( iFile ); }
				catch ( util::utilErr& e ) { cout << e.what() << endl; }
			}

			if ( !variable_defs.empty() ) // parser returned with a good result
			{
				try { // try to set the input structure in the stepwise regression object
					regressionObj.setInputStructure( variable_defs );
					variables_specified = true;
				}
				catch ( RegressNet::RegressNetErr& e ) {
					cout << e.what() << endl;
					variables_specified = false;
				}
			}
		}
		else if ( choice == 2 ) // stepwise reverse regression
		{
			if ( variables_specified )
			{
				// Function in regression.h
				try { regressionObj.reverse_regress(); }
				catch ( RegressNet::RegressNetErr& e )
					{ cout << e.what() << endl; }
			}
			else
				cout << "You must specify variables first." << endl;
		}
		else if ( choice == 3 ) // stepwise forward regression
		{
			if ( variables_specified )
			{
				// Function in regression.h
				try { regressionObj.forward_regress(); }
				catch ( RegressNet::RegressNetErr& e )
					{ cout << e.what() << endl; }
			}
			else
				cout << "You must specify variables first." << endl;
		}
	}
}

// Main menu choice 4: DFA
void dfa( DataSet* dataPtr )
{
	// DFA objects
	LDFA ldfaObj; // linear discriminant function analysis
	QDFA qdfaObj; // quadratic discriminant function analysis

	unsigned choice = 0; // menu choice

	string reply; // for alpha replies

	// Submenu loop
	while ( choice != 5 )
	{
		// Print submenu
		cout << endl << "( 1 ) Linear discriminant function analysis" << endl;
		cout << "( 2 ) Quadratic discriminant function analysis" << endl;
		cout << "( 3 ) Log last operation to file: ";
		if ( logLastOp )
			cout << "yes" << endl;
		else
			cout << "no" << endl;
		cout << "( 4 ) Log to history file: ";
		if ( logHistory )
			cout << "yes" << endl;
		else
			cout << "no" << endl;

		cout << "( 5 ) Return to main menu" << endl;

		// Query for choice
		choice = util::askI( "Menu choice: ", 1, 5 );

		// Do choice
		if ( choice == 1 ) // LDFA
		{
			ldfaObj.setDataSet( *dataPtr ); // load the dataset
			ldfaObj.setLastop( logLastOp ); // set logging options
			ldfaObj.setHistory( logHistory );
			ldfaObj.train(); // and model it with LDFA
			save_guesses( &ldfaObj ); // save model guesses if desired
		}
		else if ( choice == 2 ) // QDFA
		{
			qdfaObj.setDataSet( *dataPtr ); // load the dataset
			qdfaObj.setLastop( logLastOp ); // set logging options
			qdfaObj.setHistory( logHistory );
			qdfaObj.train(); // and model it with QDFA
			save_guesses( &qdfaObj ); // save model guesses if desired
		}
		else if ( choice == 3 ) // log last operation to file
			logLastOp = util::askYesNo( "Would you like to log the last operation to a file (yes/no)? " );

		else if ( choice == 4 ) // log to history file
			logHistory = util::askYesNo( "Would you like to log to a history file (yes/no)? " );
	}
}

// Utility function to save guesses for a model with 1 output, pass pointer to model
void save_guesses( Model* modelPtr )
{
	bool yesFlag; // for yes/no questions
	string reply; // for alpha replies

	// If it's a 1-output, discrete model, check to see if user want to save guesses
	if ( ( modelPtr->getDataSet().getOutput() == 1 ) && modelPtr->getDataSet().getDiscrete() )
	{
		yesFlag = util::askYesNo( "Would you like to save this model's training set's guesses (yes/no)? " );
		if ( yesFlag )
		{
			cout << "What is the name of the file to save the network's guesses for the training set? ";
			cin >> reply;
			modelPtr->getDataSet().saveTrainTwoSet( reply ); // save the training TwoSet object
		}

		// Query to see if the user wants to save guesses for the test set
		if ( modelPtr->getDataSet().testLoaded() )
		{
			yesFlag = util::askYesNo( "Would you like to save this model's test set's guesses (yes/no)? " );
			if ( yesFlag )
			{
				cout << "What is the name of the file to save the network's guesses for the test set? ";
				cin >> reply;
				modelPtr->getDataSet().saveTestTwoSet( reply ); // save the test TwoSet object
			}
		}
	}
}

