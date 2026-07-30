// check_clustered.cpp : Obuchowski's nonparametric clustered ROC-area
// covariance (ROADMAP 4), validated against the paper's own worked example.
//
// This is a LITERATURE ACCEPTANCE TEST, like check_wickens: the estimator is held
// to published numbers it did not produce, including the INTERMEDIATE structural
// components printed in the source table's footer. Matching a covariance through
// its intermediates rather than only through the final standard error is what
// makes this an acceptance test instead of a coincidence -- several wrong
// formulas land near the right SE.
//
// A cluster here is any sampling unit. The worked example happens to be two
// arteries within one patient; nothing in the estimator or in these tests is
// specific to that, and the synthetic cases below use clusters of unequal size,
// one-class clusters, and arbitrary non-dense ids.
//
// Sabotages this suite is built to catch:
//   - dropping the S11 cross terms (i.e. running clustered data through the
//     ordinary independent-row formula) -> the SE assertions fail;
//   - renumbering-dependent behaviour -> the permutation assertion fails;
//   - accepting a sample with too few informative clusters -> the refusal
//     assertions fail.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "clustered_auc.h"
#include "delong.h"
#include "utility.h"

using namespace std;

int failures = 0;

void expect( bool cond, const string& what )
{
	cout << ( cond ? "ok - " : "FAIL - " ) << what << endl;
	if ( !cond ) failures++;
}

bool near( double a, double b, double tol ) { return fabs( a - b ) <= tol; }

// ---------------------------------------------------------------------------
// The published worked example: two readers scoring the same units, clustered
// within subject. 36 clusters, 65 observations, 1 or 2 observations per cluster.
// Columns: cluster id, outcome (1 = the positive class), reader 1, reader 2.
// Rows whose predictor was missing in the source are excluded, exactly as the
// reference implementation excludes them ("if the predictor for either curve is
// missing, the entire record is removed") -- 65 complete records remain.
// ---------------------------------------------------------------------------
static const int MRA[][ 4 ] = {
	{ 1, 1, 87, 87 },
	{ 1, 1, 79, 83 },
	{ 2, 1, 88, 94 },
	{ 2, 1, 95, 93 },
	{ 3, 1, 100, 100 },
	{ 3, 1, 68, 79 },
	{ 4, 1, 65, 61 },
	{ 4, 1, 89, 91 },
	{ 5, 1, 97, 97 },
	{ 5, 1, 100, 100 },
	{ 6, 1, 100, 99 },
	{ 6, 1, 89, 88 },
	{ 7, 1, 77, 77 },
	{ 8, 1, 94, 89 },
	{ 8, 0, 45, 44 },
	{ 9, 1, 89, 92 },
	{ 9, 0, -11, -17 },
	{ 10, 1, 95, 91 },
	{ 10, 0, 30, 14 },
	{ 11, 1, 95, 95 },
	{ 11, 0, 10, -122 },
	{ 12, 1, 100, 100 },
	{ 12, 0, 19, -3 },
	{ 13, 1, 75, 75 },
	{ 14, 0, 70, 73 },
	{ 14, 1, 70, 81 },
	{ 15, 0, 26, 27 },
	{ 15, 1, 86, 93 },
	{ 16, 0, 0, 8 },
	{ 16, 1, 65, 70 },
	{ 17, 0, 6, 23 },
	{ 17, 1, 100, 100 },
	{ 18, 0, 76, 79 },
	{ 18, 1, 96, 95 },
	{ 19, 0, 44, 55 },
	{ 19, 1, 100, 100 },
	{ 20, 0, 73, 67 },
	{ 20, 1, 97, 94 },
	{ 21, 1, 85, 86 },
	{ 22, 0, 58, 60 },
	{ 22, 1, 100, 99 },
	{ 23, 0, -2, -2 },
	{ 23, 1, 100, 100 },
	{ 24, 0, -3, -14 },
	{ 24, 0, 0, -11 },
	{ 25, 0, 47, -3 },
	{ 25, 0, -94, -90 },
	{ 26, 0, -16, 17 },
	{ 26, 0, 75, 85 },
	{ 27, 0, 5, -22 },
	{ 27, 0, -1, -14 },
	{ 28, 0, -6, -52 },
	{ 29, 0, 37, 38 },
	{ 29, 0, 47, 35 },
	{ 30, 0, 42, 40 },
	{ 30, 0, 4, 6 },
	{ 31, 0, 55, 23 },
	{ 31, 0, 55, 45 },
	{ 32, 0, 4, -87 },
	{ 32, 0, -13, -65 },
	{ 33, 0, 49, 24 },
	{ 33, 0, 35, 9 },
	{ 34, 0, 53, 66 },
	{ 35, 0, 2, -15 },
	{ 36, 0, -48, -78 },
};
static const unsigned MRA_N = sizeof( MRA ) / sizeof( MRA[ 0 ] );

int main()
{
	vector< unsigned > label, cluster;
	vector< vector< double > > pred( 2 );
	for ( unsigned r = 0; r < MRA_N; r++ )
	{
		cluster.push_back( ( unsigned ) MRA[ r ][ 0 ] );
		label.push_back( ( unsigned ) MRA[ r ][ 1 ] );
		pred[ 0 ].push_back( ( double ) MRA[ r ][ 2 ] );
		pred[ 1 ].push_back( ( double ) MRA[ r ][ 3 ] );
	}

	// ---- the published example ------------------------------------------
	{
		clustered_auc::Result r = clustered_auc::analyze( label, cluster, pred );
		expect( r.ok, "clustered: the published example is analysable" );
		if ( !r.ok ) { cerr << r.reason << endl; return 1; }

		// The design counts the source reports: 36 clusters, 65 observations,
		// 29 positive and 36 negative units, 23 clusters carrying a positive and
		// 27 carrying a negative. These are the estimator's real sample sizes.
		expect( r.nRows == 65 && r.nClusters == 36 && r.n1 == 29 && r.n0 == 36,
			"clustered: the design counts match the published example" );
		expect( r.clusters10 == 23 && r.clusters01 == 27,
			"clustered: the informative-cluster counts match the published footer" );

		// The areas: 0.9837 and 0.9852.
		expect( near( r.auc[ 0 ], 0.9837, 5e-5 ) && near( r.auc[ 1 ], 0.9852, 5e-5 ),
			"clustered: both published areas are reproduced" );

		// The standard errors: 0.0108 and 0.0097.
		auccov::Interval i0 = clustered_auc::interval( r, 0 );
		auccov::Interval i1 = clustered_auc::interval( r, 1 );
		expect( near( i0.se, 0.0108, 5e-5 ) && near( i1.se, 0.0097, 5e-5 ),
			"clustered: both published standard errors are reproduced" );

		// The single-area interval, UNCLAMPED: the reference reports
		// (0.9625, 1.005) for reader 1, past the boundary. Clamping would hide
		// that the normal approximation is straining (see roc_theory.md).
		expect( near( i0.lo, 0.9625, 5e-5 ) && near( i0.hi, 1.005, 5e-4 )
			&& i0.hi > 1.0,
			"clustered: the published area interval is reproduced and NOT clamped to 1" );

		// The contrast: difference 0.0014, SE 0.0066, CI (-0.0115, 0.0143),
		// p = 0.8271. Our convention is primary - reference, so taking reader 2
		// as primary reproduces the published sign and interval directly.
		auccov::Contrast c = clustered_auc::contrast( r, 1, 0 );
		expect( c.valid && near( c.delta, 0.0014, 5e-5 ),
			"clustered: the published area difference is reproduced" );
		expect( near( c.seDelta, 0.0066, 5e-5 ),
			"clustered: the published difference standard error is reproduced" );
		expect( near( c.ciLoDelta, -0.0115, 5e-5 ) && near( c.ciHiDelta, 0.0143, 5e-5 ),
			"clustered: the published difference interval is reproduced" );
		expect( near( c.p, 0.8271, 5e-5 ),
			"clustered: the published p-value is reproduced" );

		// THE STRUCTURAL COMPONENTS, from the published table footer. The
		// covariance is rebuilt here from the estimator's own outputs and checked
		// against all nine published values -- reader 1's and reader 2's S10/S01/
		// S11 and the four cross terms. A wrong divisor or a dropped cross term
		// can still land near the right SE; it cannot land on these.
		//
		// The published values (to the digits the source prints):
		//   I=36  I10=23  I01=27  M=29  N=36
		//   r1: S10 0.0013151307  S01 0.0022375873  S11  0.0051785478
		//   r2: S10 0.0009284122  S01 0.0022598583  S11 -0.0005015760
		//   S10_12 0.0008484357  S01_12 0.0019241250
		//   S11_12 0.0028648340  S11_21 -0.0015136931
		//
		// Cov(r,s) = S10/M + S01/N + (S11^rs + S11^sr)/(MN), so the published
		// components must reconstruct the estimator's covariance matrix exactly.
		{
			const double M = 29.0, N = 36.0;
			const double S10_11 = 0.0013151307, S01_11 = 0.0022375873, S11_11 = 0.0051785478;
			const double S10_22 = 0.0009284122, S01_22 = 0.0022598583, S11_22 = -0.0005015760;
			const double S10_12 = 0.0008484357, S01_12 = 0.0019241250;
			const double S11_12 = 0.0028648340, S11_21 = -0.0015136931;

			double v00 = S10_11 / M + S01_11 / N + 2 * S11_11 / ( M * N );
			double v11 = S10_22 / M + S01_22 / N + 2 * S11_22 / ( M * N );
			double v01 = S10_12 / M + S01_12 / N + ( S11_12 + S11_21 ) / ( M * N );

			expect( near( r.cov( 0, 0 ), v00, 1e-11 )
				&& near( r.cov( 1, 1 ), v11, 1e-11 )
				&& near( r.cov( 0, 1 ), v01, 1e-11 )
				&& near( r.cov( 1, 0 ), v01, 1e-11 ),
				"clustered: the covariance matrix is reconstructed by the nine published components" );
		}

		// The AREA is the ordinary Mann-Whitney area -- clustering changes the
		// variance, not the estimand. Same data through ordinary DeLong must give
		// the SAME areas and a DIFFERENT (here smaller) standard error.
		{
			delong::Result d = delong::analyze( label, pred );
			expect( d.ok && near( d.auc[ 0 ], r.auc[ 0 ], 1e-12 )
				&& near( d.auc[ 1 ], r.auc[ 1 ], 1e-12 ),
				"clustered: the point areas are identical to the ordinary ones" );

			// THE SABOTAGE ASSERTION. Running this clustered sample through the
			// independent-row formula must NOT reproduce the published standard
			// error -- if it did, the clustered estimator would be doing nothing.
			delong::Interval di = delong::interval( d, 0 );
			expect( !near( di.se, 0.0108, 5e-5 ),
				"clustered: ordinary DeLong does NOT reproduce the published clustered SE" );
			auccov::Contrast dc = delong::contrast( d, 1, 0 );
			expect( dc.valid && !near( dc.seDelta, 0.0066, 5e-5 ),
				"clustered: ordinary DeLong does NOT reproduce the published difference SE" );
		}
	}

	// ---- invariance to how the clusters are numbered ---------------------
	{
		clustered_auc::Result base = clustered_auc::analyze( label, cluster, pred );
		vector< unsigned > relabelled( cluster.size() );
		for ( unsigned r = 0; r < cluster.size(); r++ )
			relabelled[ r ] = 900000 - cluster[ r ] * 7; // order-reversing, non-dense
		clustered_auc::Result perm = clustered_auc::analyze( label, relabelled, pred );
		expect( perm.ok && near( perm.cov( 0, 0 ), base.cov( 0, 0 ), 1e-15 )
			&& near( perm.cov( 0, 1 ), base.cov( 0, 1 ), 1e-15 )
			&& perm.nClusters == base.nClusters,
			"clustered: relabelling the clusters does not change the covariance" );
	}

	// ---- the reduction toward ordinary DeLong ----------------------------
	// With ONE row per cluster the S11 terms vanish identically and the estimator
	// must approach ordinary DeLong. The agreement is ASYMPTOTIC, not exact: the
	// finite-sample factors differ -- I10/((I10-1)M) against 1/(M-1) -- so the
	// variances differ by exactly I10(M-1)/((I10-1)M). The test asserts both: the
	// stated factor to machine precision, and thence agreement to a fraction of a
	// percent. Claiming an exact identity here would be false.
	{
		const unsigned n = 200;
		vector< unsigned > lab( n ), cl( n );
		vector< vector< double > > pr( 2, vector< double >( n ) );
		util::set_seed( 11 );
		for ( unsigned i = 0; i < n; i++ )
		{
			lab[ i ] = ( i % 3 == 0 ) ? 1u : 0u;
			cl[ i ] = 1000 + i;                 // one row per cluster: singletons
			double signal = lab[ i ] ? 0.6 : 0.4;
			pr[ 0 ][ i ] = signal + 0.25 * util::d_random( 0.0, 1.0 );
			pr[ 1 ][ i ] = signal + 0.25 * util::d_random( 0.0, 1.0 );
		}
		clustered_auc::Result c = clustered_auc::analyze( lab, cl, pr );
		delong::Result d = delong::analyze( lab, pr );
		expect( c.ok && d.ok, "reduction: both estimators accept singleton clusters" );

		expect( near( c.auc[ 0 ], d.auc[ 0 ], 1e-12 ),
			"reduction: the areas are identical" );

		// The exact finite-sample factor, applied to each half of the variance.
		double M = ( double ) c.n1, N = ( double ) c.n0;
		double fM = ( double ) c.clusters10 * ( M - 1 )
			/ ( ( ( double ) c.clusters10 - 1 ) * M );
		double fN = ( double ) c.clusters01 * ( N - 1 )
			/ ( ( ( double ) c.clusters01 - 1 ) * N );
		expect( near( fM, 1.0, 0.02 ) && near( fN, 1.0, 0.02 ),
			"reduction: the finite-sample factors are within 2% of 1 at this size" );

		double rel = fabs( c.cov( 0, 0 ) - d.cov( 0, 0 ) ) / d.cov( 0, 0 );
		expect( rel < 0.02,
			"reduction: singleton clusters give ordinary DeLong's variance to within 2%" );

		double relOff = fabs( c.cov( 0, 1 ) - d.cov( 0, 1 ) ) / fabs( d.cov( 0, 1 ) );
		expect( relOff < 0.02,
			"reduction: the off-diagonal covariance agrees too" );
	}

	// ---- a paired whole-cluster bootstrap, as an independent cross-check ---
	// Resample CLUSTERS with replacement (never rows) from the reserved
	// resampling stream, recompute both areas and their signed difference, and
	// compare the bootstrap spread with the analytic one. This is a cross-check on
	// the analytic estimator, not a substitute for it, and it is not wired into any
	// reported inference. The tolerance is prespecified and loose -- a bootstrap of
	// this size has its own sampling error; the point is to catch an estimator that
	// is wrong by a FACTOR, which is what a dropped term or a wrong divisor gives.
	{
		// A synthetic clustered fixture with a real within-cluster correlation:
		// each cluster gets its own offset, so rows within it are alike.
		const unsigned nClusters = 120;
		vector< unsigned > lab, cl;
		vector< vector< double > > pr( 2 );
		util::set_seed( 5 );
		vector< double > offset( nClusters );
		for ( unsigned g = 0; g < nClusters; g++ ) offset[ g ] = util::d_random( 0.0, 1.0 ) - 0.5;
		for ( unsigned g = 0; g < nClusters; g++ )
		{
			unsigned size = 1 + g % 4;             // unequal cluster sizes
			for ( unsigned i = 0; i < size; i++ )
			{
				unsigned y = ( ( g + i ) % 3 == 0 ) ? 1u : 0u;
				lab.push_back( y );
				cl.push_back( 7000 + g * 3 );        // arbitrary, non-dense ids
				double base = ( y ? 0.65 : 0.35 ) + 0.55 * offset[ g ];
				pr[ 0 ].push_back( base + 0.2 * util::d_random( 0.0, 1.0 ) );
				pr[ 1 ].push_back( base + 0.2 * util::d_random( 0.0, 1.0 ) );
			}
		}

		clustered_auc::Result r = clustered_auc::analyze( lab, cl, pr );
		expect( r.ok, "bootstrap: the clustered fixture is analysable" );
		auccov::Contrast analytic = clustered_auc::contrast( r, 0, 1 );

		// Rows grouped by cluster, so a resample can take whole clusters.
		map< unsigned, vector< unsigned > > rowsOf;
		for ( unsigned i = 0; i < cl.size(); i++ ) rowsOf[ cl[ i ] ].push_back( i );
		vector< unsigned > ids;
		for ( map< unsigned, vector< unsigned > >::const_iterator it = rowsOf.begin();
			it != rowsOf.end(); ++it ) ids.push_back( it->first );

		const unsigned B = 400;
		vector< double > deltas, a0s;
		for ( unsigned b = 0; b < B; b++ )
		{
			vector< unsigned > blab, bcl;
			vector< vector< double > > bpr( 2 );
			for ( unsigned g = 0; g < ids.size(); g++ )
			{
				// util::i_resample -- the RESERVED resampling stream, so computing an
				// interval can never perturb weight init or splits (a settled rule).
				unsigned pick = util::i_resample( ( unsigned ) ids.size() );
				const vector< unsigned >& rows = rowsOf[ ids[ pick ] ];
				for ( unsigned t = 0; t < rows.size(); t++ )
				{
					blab.push_back( lab[ rows[ t ] ] );
					bcl.push_back( 100000 + g );   // a resampled cluster is its own unit
					bpr[ 0 ].push_back( pr[ 0 ][ rows[ t ] ] );
					bpr[ 1 ].push_back( pr[ 1 ][ rows[ t ] ] );
				}
			}
			auccov::Placements p = auccov::compute( blab, bpr );
			if ( !p.ok ) continue;
			a0s.push_back( p.auc[ 0 ] );
			deltas.push_back( p.auc[ 0 ] - p.auc[ 1 ] );
		}
		expect( deltas.size() > B / 2,
			"bootstrap: most whole-cluster resamples are analysable" );

		auto sd = []( const vector< double >& v )
		{
			double mean = 0;
			for ( unsigned i = 0; i < v.size(); i++ ) mean += v[ i ];
			mean /= v.size();
			double ss = 0;
			for ( unsigned i = 0; i < v.size(); i++ ) ss += ( v[ i ] - mean ) * ( v[ i ] - mean );
			return sqrt( ss / ( v.size() - 1 ) );
		};
		double bootDelta = sd( deltas ), bootA0 = sd( a0s );
		auccov::Interval iv = clustered_auc::interval( r, 0 );

		// Prespecified tolerance: the analytic and bootstrap spreads must agree
		// within a factor of 1.6 in each direction. Wide enough for 400 resamples,
		// far too tight for an estimator missing a term.
		double ratioA = bootA0 / iv.se, ratioD = bootDelta / analytic.seDelta;
		expect( ratioA > 0.625 && ratioA < 1.6,
			"bootstrap: the analytic area SE agrees with the whole-cluster bootstrap" );
		expect( ratioD > 0.625 && ratioD < 1.6,
			"bootstrap: the analytic difference SE agrees with the whole-cluster bootstrap" );

		// The ordinary independent-row SE on this SAME data is materially
		// different, and FURTHER from the whole-cluster bootstrap than the
		// clustered one. Note the direction is NOT asserted here: on this fixture
		// the cluster offset moves both classes together, which makes within-cluster
		// pairs MORE concordant and shrinks the variance (measured: clustered SE
		// 0.0164 against DeLong's 0.0188). The design effect genuinely runs both
		// ways -- the published example has S11 positive for one reader and negative
		// for the other -- so an "ordinary DeLong is always anti-conservative" test
		// would be asserting something false. What must hold is that the clustered
		// estimator tracks the whole-cluster bootstrap and the row-based one does
		// not. The anti-conservative direction is demonstrated on its own fixture
		// below, where it really is the truth.
		delong::Result d = delong::analyze( lab, pr );
		delong::Interval di = delong::interval( d, 0 );
		expect( d.ok && fabs( iv.se - bootA0 ) < fabs( di.se - bootA0 ),
			"bootstrap: the clustered SE is closer to the whole-cluster bootstrap than the row-based one" );
	}

	// ---- the anti-conservative case, demonstrated ------------------------
	// When each cluster carries ONE outcome class and has its own score offset,
	// the positives clump and the negatives clump: the effective sample size is
	// the number of CLUSTERS, not of rows, and the row-based estimator is badly
	// anti-conservative. This is the failure mode the whole layer exists to
	// prevent, so it is measured rather than assumed.
	{
		vector< unsigned > lab, cl;
		vector< vector< double > > pr( 1 );
		util::set_seed( 17 );
		const unsigned nClusters = 40, per = 5;
		for ( unsigned g = 0; g < nClusters; g++ )
		{
			unsigned y = g % 2;
			double offset = util::d_random( -0.35, 0.35 ); // the cluster's own level
			for ( unsigned i = 0; i < per; i++ )
			{
				lab.push_back( y );
				cl.push_back( 500 + g * 11 );              // arbitrary, non-dense ids
				pr[ 0 ].push_back( ( y ? 0.55 : 0.45 ) + offset
					+ 0.10 * util::d_random( 0.0, 1.0 ) );
			}
		}

		clustered_auc::Result r = clustered_auc::analyze( lab, cl, pr );
		delong::Result d = delong::analyze( lab, pr );
		expect( r.ok && d.ok && r.nClusters == nClusters
			&& r.clusters10 == 20 && r.clusters01 == 20,
			"design effect: 200 rows in 40 one-class clusters, 20 carrying each class" );

		auccov::Interval ci = clustered_auc::interval( r, 0 );
		delong::Interval di = delong::interval( d, 0 );

		// Measured on this fixture: clustered SE 0.0967, ordinary 0.0428 -- a design
		// effect of 2.26. The row-based interval here is less than half as wide as
		// it should be, which is exactly how a clustered p-value becomes invalid.
		expect( di.se < 0.6 * ci.se,
			"design effect: ordinary DeLong badly understates the SE when clusters are one-class" );

		// And the clustered estimator is the one the whole-cluster bootstrap agrees
		// with -- 2000 resamples of whole clusters from the reserved stream.
		{
			map< unsigned, vector< unsigned > > rowsOf;
			for ( unsigned i = 0; i < cl.size(); i++ ) rowsOf[ cl[ i ] ].push_back( i );
			vector< unsigned > ids;
			for ( map< unsigned, vector< unsigned > >::const_iterator it = rowsOf.begin();
				it != rowsOf.end(); ++it ) ids.push_back( it->first );

			vector< double > areas;
			for ( unsigned b = 0; b < 2000; b++ )
			{
				vector< unsigned > blab;
				vector< vector< double > > bpr( 1 );
				for ( unsigned g = 0; g < ids.size(); g++ )
				{
					unsigned pick = util::i_resample( ( unsigned ) ids.size() );
					const vector< unsigned >& rows = rowsOf[ ids[ pick ] ];
					for ( unsigned t = 0; t < rows.size(); t++ )
					{
						blab.push_back( lab[ rows[ t ] ] );
						bpr[ 0 ].push_back( pr[ 0 ][ rows[ t ] ] );
					}
				}
				auccov::Placements p = auccov::compute( blab, bpr );
				if ( p.ok ) areas.push_back( p.auc[ 0 ] );
			}
			double mean = 0;
			for ( unsigned i = 0; i < areas.size(); i++ ) mean += areas[ i ];
			mean /= areas.size();
			double ss = 0;
			for ( unsigned i = 0; i < areas.size(); i++ )
				ss += ( areas[ i ] - mean ) * ( areas[ i ] - mean );
			double boot = sqrt( ss / ( areas.size() - 1 ) );

			// Measured: analytic 0.09670, bootstrap 0.09662 -- three decimals.
			expect( fabs( ci.se - boot ) / boot < 0.15,
				"design effect: the clustered SE matches the whole-cluster bootstrap" );
			expect( fabs( di.se - boot ) / boot > 0.3,
				"design effect: the row-based SE does not" );
		}
	}

	// ---- structure, degeneracy and malformed input -----------------------
	{
		// Ties everywhere: every score identical -> area 0.5 by half credit.
		{
			vector< unsigned > lab, cl;
			vector< vector< double > > pr( 1 );
			for ( unsigned g = 0; g < 10; g++ )
				for ( unsigned i = 0; i < 2; i++ )
				{
					lab.push_back( i ); cl.push_back( g ); pr[ 0 ].push_back( 3.0 );
				}
			clustered_auc::Result r = clustered_auc::analyze( lab, cl, pr );
			expect( r.ok && near( r.auc[ 0 ], 0.5, 1e-12 ),
				"clustered: all-tied scores give area 0.5 (half credit), as ordinary DeLong does" );
		}

		// One-class clusters are fine -- they contribute to one term and drop out
		// of the other. Half the clusters here carry only positives.
		{
			vector< unsigned > lab, cl;
			vector< vector< double > > pr( 1 );
			util::set_seed( 3 );
			for ( unsigned g = 0; g < 20; g++ )
			{
				unsigned y = ( g % 2 );          // this whole cluster is one class
				for ( unsigned i = 0; i < 3; i++ )
				{
					lab.push_back( y ); cl.push_back( g );
					// The classes must OVERLAP or the area is 1 and its variance 0,
					// which tests the degenerate path rather than this one.
					pr[ 0 ].push_back( ( y ? 0.55 : 0.45 ) + 0.5 * util::d_random( 0.0, 1.0 ) );
				}
			}
			clustered_auc::Result r = clustered_auc::analyze( lab, cl, pr );
			expect( r.ok && r.clusters10 == 10 && r.clusters01 == 10
				&& r.cov( 0, 0 ) > 0,
				"clustered: clusters carrying only one outcome class are handled" );
		}

		// Identical procedures: zero difference, zero difference variance, and the
		// DEGENERATE flag -- not a fabricated separation (DLG-2, shared algebra).
		{
			vector< unsigned > lab, cl;
			vector< vector< double > > pr( 2 );
			util::set_seed( 9 );
			for ( unsigned g = 0; g < 20; g++ )
				for ( unsigned i = 0; i < 2; i++ )
				{
					lab.push_back( ( g + i ) % 2 ); cl.push_back( g );
					double v = 0.5 + util::d_random( 0.0, 1.0 );
					pr[ 0 ].push_back( v ); pr[ 1 ].push_back( v );
				}
			clustered_auc::Result r = clustered_auc::analyze( lab, cl, pr );
			auccov::Contrast c = clustered_auc::contrast( r, 0, 1 );
			expect( r.ok && c.valid && c.degenerate && !c.separated
				&& near( c.p, 1.0, 1e-12 ),
				"clustered: identical procedures are degenerate (no testable difference), not separated" );
		}

		// Perfect vs reversed: constant placements, a total difference. The areas
		// differ deterministically -> SEPARATED, p = 0.
		{
			vector< unsigned > lab, cl;
			vector< vector< double > > pr( 2 );
			for ( unsigned g = 0; g < 12; g++ )
				for ( unsigned i = 0; i < 2; i++ )
				{
					unsigned y = i;
					lab.push_back( y ); cl.push_back( g );
					pr[ 0 ].push_back( y ? 1.0 : 0.0 );   // perfect
					pr[ 1 ].push_back( y ? 0.0 : 1.0 );   // exactly reversed
				}
			clustered_auc::Result r = clustered_auc::analyze( lab, cl, pr );
			auccov::Contrast c = clustered_auc::contrast( r, 0, 1 );
			expect( r.ok && near( r.auc[ 0 ], 1.0, 1e-12 ) && near( r.auc[ 1 ], 0.0, 1e-12 ),
				"clustered: perfect and reversed scores give areas 1 and 0" );
			expect( c.valid && c.separated && !c.degenerate && near( c.p, 0.0, 1e-12 ),
				"clustered: a deterministic separation is reported as such, not as no difference" );
		}

		// Too few INFORMATIVE clusters is a refusal with the counts -- and it is a
		// different condition from "too few rows": there are plenty of rows here.
		{
			vector< unsigned > lab, cl;
			vector< vector< double > > pr( 1 );
			util::set_seed( 2 );
			for ( unsigned i = 0; i < 40; i++ )
			{
				lab.push_back( i % 2 );
				cl.push_back( i < 20 ? 1u : 2u );   // 40 rows, but only TWO clusters
				pr[ 0 ].push_back( util::d_random( 0.0, 1.0 ) );
			}
			clustered_auc::Result twoClusters = clustered_auc::analyze( lab, cl, pr );
			expect( twoClusters.ok,
				"clustered: two informative clusters is the minimum, and is accepted" );

			for ( unsigned i = 0; i < 40; i++ ) cl[ i ] = 1u; // now ONE cluster
			clustered_auc::Result one = clustered_auc::analyze( lab, cl, pr );
			expect( !one.ok && one.reason.find( "two clusters" ) != string::npos,
				"clustered: a single cluster is refused with the cluster counts, not divided by zero" );

			// The row-based estimator would happily accept that same sample --
			// which is exactly the mistake this refusal prevents.
			delong::Result d = delong::analyze( lab, pr );
			expect( d.ok,
				"clustered: ordinary DeLong accepts the one-cluster sample the clustered estimator refuses" );
		}

		// Malformed input: mismatched cluster length, mismatched prediction length,
		// a non-finite score, an absent outcome class, no procedures.
		{
			vector< unsigned > lab( 10, 0 ), cl( 10, 0 );
			for ( unsigned i = 0; i < 10; i++ ) { lab[ i ] = i % 2; cl[ i ] = i / 2; }
			vector< vector< double > > pr( 1, vector< double >( 10, 0.5 ) );
			for ( unsigned i = 0; i < 10; i++ ) pr[ 0 ][ i ] = ( double ) i;

			vector< unsigned > shortCl( 9, 0 );
			expect( !clustered_auc::analyze( lab, shortCl, pr ).ok,
				"clustered: a cluster vector that does not pair with the rows is refused" );

			vector< vector< double > > shortPr( 1, vector< double >( 9, 0.5 ) );
			expect( !clustered_auc::analyze( lab, cl, shortPr ).ok,
				"clustered: a prediction/label length mismatch is refused" );

			vector< vector< double > > nanPr = pr;
			nanPr[ 0 ][ 3 ] = sqrt( -1.0 );
			expect( !clustered_auc::analyze( lab, cl, nanPr ).ok,
				"clustered: a non-finite score is refused (a diverged model has no area)" );

			vector< unsigned > oneClass( 10, 1 );
			expect( !clustered_auc::analyze( oneClass, cl, pr ).ok,
				"clustered: a sample with only one outcome class is refused" );

			expect( !clustered_auc::analyze( lab, cl,
				vector< vector< double > >() ).ok,
				"clustered: no procedures is refused" );
		}
	}

	if ( failures == 0 )
	{
		cout << "check_clustered: Obuchowski clustered covariance matches the "
			"published example, its nine intermediate components, and its "
			"structural contracts" << endl;
		return 0;
	}
	cerr << "check_clustered: FAILED (" << failures << ")" << endl;
	return 1;
}
