#include "procguard.h"

#include <exception>

namespace procguard
{

int run( const std::function< int() >& body, std::ostream& err,
	const char* program )
{
	try
	{
		return body();
	}
	catch ( const std::exception& e )
	{
		// Matrix and vector_ops contract failures reach here with their
		//    messages intact, which is the whole reason D9 gave them one
		err << program << ": fatal: " << e.what() << std::endl;
		return 1;
	}
	catch ( ... )
	{
		err << program << ": fatal: unrecognized error" << std::endl;
		return 1;
	}
}

} // namespace procguard
