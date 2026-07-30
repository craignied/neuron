// scale_probe.cpp : a reproducible SCALE measurement for the ROADMAP 4 partition
// and clustered-inference layer, on GENERATED data -- no external dataset needed.
//
// It is built by CI but deliberately NOT registered as a ctest case: it measures
// wall time and memory, which are machine-dependent, and a timing assertion in
// the test suite would be a flake generator. Building it keeps it from rotting;
// running it is a deliberate act.
//
//   ./build/scale_probe [rows] [clusters]      (defaults: 226679 rows, 612 clusters)
//
// Cluster sizes are deliberately SKEWED (a few large, many small) and the event
// rate varies by cluster, because an evenly sized synthetic cohort would not
// exercise the packing or the between-cluster variance at all.
//
// Measured on an Apple Silicon Mac, Release, 226,679 rows / 612 clusters /
// 2.91% events: group holdout 0.004 s, group 5-fold plan 0.003 s (zero leakage,
// imbalance 0.0023), clustered covariance over 44,068 locked rows in 172
// clusters 0.012 s, peak RSS 18 MB. Nothing in this layer is quadratic.

#include "split.h"
#include "clustered_auc.h"
#include "utility.h"
#include <chrono>
#include <cstdio>
#include <set>
#include <map>
#include <cstdlib>
using namespace std::chrono;
int main( int argc, char** argv )
{
	unsigned n = argc > 1 ? (unsigned)atoi(argv[1]) : 226679;
	unsigned G = argc > 2 ? (unsigned)atoi(argv[2]) : 612;
	util::set_seed( 42 );
	vector<unsigned> label(n), group(n);
	vector<vector<double>> pred(2, vector<double>(n));
	vector<double> off(G);
	for (unsigned g=0; g<G; g++) off[g] = util::d_random(-0.3,0.3);
	// Skewed cluster sizes: a few large, many small -- the realistic shape.
	vector<unsigned> gOf(n);
	{
		double tot=0; vector<double> w(G);
		for (unsigned g=0; g<G; g++){ w[g]=1.0/(1.0+g%97); tot+=w[g]; }
		unsigned r=0;
		for (unsigned g=0; g<G && r<n; g++){
			unsigned cnt=(unsigned)(w[g]/tot*n)+1;
			for (unsigned i=0;i<cnt && r<n;i++) gOf[r++]=g;
		}
		while (r<n) gOf[r++]=G-1;
	}
	for (unsigned r=0;r<n;r++){
		group[r]=1000+gOf[r]*7;
		label[r]=(util::d_random(0.0,1.0) < 0.0296+0.05*off[gOf[r]]) ? 1u : 0u;
		double base=(label[r]?0.6:0.4)+0.4*off[gOf[r]];
		pred[0][r]=base+0.25*util::d_random(0.0,1.0);
		pred[1][r]=base+0.25*util::d_random(0.0,1.0);
	}
	unsigned ev=0; for(unsigned r=0;r<n;r++) ev+=label[r];
	printf("generated: n=%u clusters=%u events=%u (%.2f%%)\n",n,G,ev,100.0*ev/n);

	auto t0=steady_clock::now();
	nsplit::GroupHoldout h = nsplit::groupHoldout(label, [&]{
		// densify by first appearance, as DataSet::groupKey does
		vector<unsigned> gid(n); map<unsigned,unsigned> d;
		for(unsigned r=0;r<n;r++){ auto it=d.find(group[r]);
			if(it==d.end()){ unsigned id=d.size(); d[group[r]]=id; gid[r]=id; } else gid[r]=it->second; }
		return gid; }(), n/5);
	auto t1=steady_clock::now();
	printf("group holdout      : %6.3f s  (locked %zu rows, %u groups, %u in test)\n",
		duration<double>(t1-t0).count(), h.test.size(), h.nGroups, h.groupsInTest);

	// fold plan over the development rows
	vector<unsigned> devLab, devGrp;
	for(unsigned i=0;i<h.train.size();i++){ devLab.push_back(label[h.train[i]]); devGrp.push_back(group[h.train[i]]); }
	auto t2=steady_clock::now();
	nsplit::GroupFoldPlan p = nsplit::stratifiedGroupKFold(devLab, devGrp, 5);
	auto t3=steady_clock::now();
	printf("group 5-fold plan  : %6.3f s  (ok=%d groups=%u leakage=%u imbalance=%.4f)\n",
		duration<double>(t3-t2).count(),(int)p.ok,p.nGroups,p.leakageCount,p.imbalanceScore);

	// clustered covariance on the locked sample
	vector<unsigned> lLab,lCl; vector<vector<double>> lPred(2);
	for(unsigned i=0;i<h.test.size();i++){ unsigned r=h.test[i];
		lLab.push_back(label[r]); lCl.push_back(group[r]);
		lPred[0].push_back(pred[0][r]); lPred[1].push_back(pred[1][r]); }
	auto t4=steady_clock::now();
	clustered_auc::Result cr = clustered_auc::analyze(lLab,lCl,lPred);
	auto t5=steady_clock::now();
	printf("clustered covar    : %6.3f s  (ok=%d rows=%u clusters=%u c10=%u c01=%u)\n",
		duration<double>(t5-t4).count(),(int)cr.ok,cr.nRows,cr.nClusters,cr.clusters10,cr.clusters01);
	if(cr.ok){ auccov::Interval iv=clustered_auc::interval(cr,0);
		printf("                     AUC %.4f  clustered SE %.5f\n",iv.auc,iv.se); }
	return 0;
}
