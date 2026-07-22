/* Model construction, factored to one authoritative place (rule 6 / the layer-
   ownership constitution). The CLI driver (neuron.cpp specify_model) and the GUI
   (gui.cpp handleModel) each built a model by type and wired it up inline -- the
   same "1 output + 1 layer -> SimpleProp/BareProp, else BackProp" rule and the
   same manifest construction order (setBias -> setDataSet -> error -> setHidden).
   That logic lives here now; both drivers call it and neither copies it.

   This is a model CREATION service (you have no object yet, so it cannot be
   virtual dispatch) -- distinct from a forbidden god TRAINING switch: fitting
   stays each concrete model's own virtual train(). The factory reaches only the
   models' public interfaces and preserves the construction order the manifest
   requires. It does not train, evaluate, or know about folds, HTTP, or SEER. */

#ifndef MODELFACTORY_H
#define MODELFACTORY_H

#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "dataset.h"

using namespace std;

namespace modelfactory {

// The architecture a user asked for, independent of how they asked (menu or
//    HTTP). The concrete class is DERIVED from this + the data's output count.
struct Spec {
	bool logistic = false;    // binary logistic regression (ignores the NN fields)
	bool bias = true;         // bias nodes (neural nets only)
	vector< unsigned > hidden; // hidden-layer sizes (neural nets only)
	bool xentropy = true;     // output error X-entropy (true) vs LMS (false); n/a to logistic
	bool logLastop = false;   // log the last operation to model.txt
	bool logHistory = true;   // log to neuron.log
};

// Build a fresh (untrained) model from a spec, bound to data, in the manifest's
//    construction order. The concrete type follows the CLI/GUI rule: logistic ->
//    Logistic; 1 output AND 1 hidden layer -> SimpleProp (bias) or BareProp (no
//    bias); otherwise BackProp. Returns nullptr and sets err on an invalid
//    request (logistic needs exactly 1 discrete output; a net needs >= 1 layer).
unique_ptr< Model > build( const Spec& spec, DataSet& data, string& err );

// Create an empty model of a named type, for the load-from-file path (the type
//    is line 1 of a saved network): "Binary logistic" | "SimpleProp" |
//    "BareProp" | "BackProp". bias applies to BackProp (its line-2 flag). The
//    caller then wires the dataset/flags and calls Network::load. Returns
//    nullptr for an unrecognized type.
unique_ptr< Model > createByTypeName( const string& typeName, bool bias );

} // namespace modelfactory

#endif
