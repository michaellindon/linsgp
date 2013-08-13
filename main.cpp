#include <cstdlib>
#include <iostream>
#include <vector>
#include <new>
#include <trng/yarn2.hpp>
#include <trng/normal_dist.hpp>
#include <trng/lognormal_dist.hpp>
#ifdef _OPENMP
#include <omp.h>
#else 
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

class particles{
	public:
		std::vector<double> phi,phit,K,Kt,Z,eps;

};


//Constructs Correlation matrix K using the first "n" rows of X i.e. K(X1:n)
void corrKt(particles &particle,std::vector<double> &X, int n, int Dim){

	particle.Kt.resize(n*n,0);
	particle.eps.resize(n,0);


	for (int r = 0; r < n; ++r)
	{
		for (int c = 0; c <= r; ++c)
		{
			for (int d = 0; d < Dim; ++d)
			{
				particle.Kt[r*n+c]+=particle.phit[d]*(X[r*n + d] - X[c*n + d])*(X[r*n + d] - X[c*n + d]);
			}

			particle.Kt[r*n+c]=exp(-particle.Kt[r*n+c]);
			particle.Kt[c*n+r]=particle.Kt[r*n+c];

		}
	}
}


using namespace std;

extern "C" void linsgp(double *XR, int *X_nrow, int *X_ncol, int *np, int *niter, double *mphi, double *mphit, double *vphi, double *vphit)
{
	int i,d;
	int nobs=*X_nrow; //nobs ~ number of observations
	int Dim=*X_ncol; //Dim ~ dimension of Domain, excluding latent variable.
	int nthreads=1; //nthreads ~ number of threads. 1 by default.
	int rank=0; //thread identifier. 0 by default.

	//Convert C-style array of doubles to C++-style vector of doubles
	std::vector<double> X(*X_ncol * *X_nrow);
	std::copy (XR, XR+(*X_nrow * *X_ncol), X.begin() );

	particles * particle; 
	trng::normal_dist<> normal(0.0,1.0);
	particle = new (nothrow)  particles[*np]; //Create Particles

	//Check if memory allocation was successful
	if (particle==0){
		std::cout << "Dynamic memory allocation for " 
			<< *np
			<< " particles was not successful." 
			<< std::endl;
		std::cout << "Please consider using fewer particles." 
			<< std::endl;
		//	return EXIT_FAILURE;
	}


	//Query maximum threads supported by hardware
	nthreads=omp_get_max_threads();

	//Set number of threads to the supported maximum
#ifdef _OPENMP
	omp_set_num_threads(nthreads);
#endif
	//Create "nthreads" number of independent random number streams 
	trng::yarn2 * stream;
	stream=new (nothrow) trng::yarn2[nthreads];
	if(stream==0){
		std::cout << "Dynamic memory allocation for "
			<< nthreads
			<< " random number streams failed."
			<<std::endl ;
	}


	for (i = 0; i < nthreads; i++)
	{
		stream[i].split(nthreads,i);
	}



//*******INITIALIZATION*********//
	trng::lognormal_dist<> lognormal_phi(*mphi,*vphi);
	trng::lognormal_dist<> lognormal_phit(*mphit,*vphit);
	trng::normal_dist<> standard_normal(0,1);
#pragma omp parallel for private(rank,i,d) 
	for (int p = 0; p < *np; ++p)  //p for looping over particles
	{
		rank=omp_get_thread_num();
		particle[p].phi.resize(Dim+1,0);
		particle[p].phit.resize(Dim,0);
		particle[p].K.reserve(nobs * nobs);
		particle[p].Kt.reserve(nobs * nobs);
		particle[p].eps.reserve(nobs);


		for (d = 0; d < Dim; ++d) //d for looping over dimension
		{
			particle[p].phi[d] = lognormal_phi(stream[rank]);
			particle[p].phit[d] = lognormal_phi(stream[rank]);
		}
		particle[p].phi[Dim] = lognormal_phi(stream[rank]);

		corrKt(particle[p],X,5,Dim);

		for(unsigned int t=0; t<particle[p].eps.size() ; ++t){
			particle[p].eps[t]=standard_normal(stream[rank]);
		}




	}
//******END INITIALIZATION********//


	for (unsigned int i = 0; i < particle[1].Kt.size(); ++i)
	{
		cout << particle[1].Kt[i] << endl;
	}
cout << "hello" << endl;
	for (unsigned int i = 0; i < particle[1].eps.size(); ++i)
	{
		cout << particle[1].eps[i] << endl;
	}

	delete[] particle; //Free memory
	delete[] stream;

	//	return 0;
}


