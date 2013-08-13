#include <cstdlib>
#include <iostream>
#include <vector>
#include <new>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
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
		std::vector<double> phi,phit,K,Kt,Z;

};


//Constructs Correlation matrix K using the first "n" rows of X i.e. K(X1:n)
void corr(particles &particle,std::vector<double> &X, int n, int Dim){
	int i,j,d;

	particle.K.resize(n*n,0);
	particle.Kt.resize(n*n,0);
	for (i = 0; i < n; ++i)
	{
		for (j = 0; j <= i; ++j)
		{
			for (d = 0; d < Dim; ++d)
			{
				particle.K[i*n+j]+=particle.phi[d]*(X[i*n + d] - X[j*n + d])*(X[i*n + d] - X[j*n + d]);
				particle.Kt[i*n+j]+=particle.phit[d]*(X[i*n + d] - X[j*n + d])*(X[i*n + d] - X[j*n + d]);
			}
			//particle.K[Dim]+=particle.phi[Dim]*(particle.Z[i]-particle.Z[j])*(particle.Z[i] - particle.Z[j]);

			particle.K[i*n+j]=exp(-particle.K[i*n+j]);
			particle.K[j*n+i]=particle.K[i*n+j];

			particle.Kt[i*n+j]=exp(-particle.Kt[i*n+j]);
			particle.Kt[j*n+i]=particle.Kt[i*n+j];

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



	//Initialize particles with draws from the posterior
	trng::lognormal_dist<> lognormal_phi(*mphi,*vphi);
	trng::lognormal_dist<> lognormal_phit(*mphit,*vphit);
#pragma omp parallel for private(rank,i,d) 
	for (i = 0; i < *np; ++i)
	{
		rank=omp_get_thread_num();
		particle[i].phi.resize(Dim+1,0);
		particle[i].phit.resize(Dim,0);
		particle[i].K.reserve(nobs * nobs);
		particle[i].Kt.reserve(nobs * nobs);

		for (d = 0; d < Dim; ++d)
		{
			particle[i].phi[d] = lognormal_phi(stream[rank]);
			particle[i].phit[d] = lognormal_phi(stream[rank]);
		}
		particle[i].phi[Dim] = lognormal_phi(stream[rank]);

		corr(particle[i],X,5,Dim);
	}


	for (int i = 0; (unsigned int) i < particle[1].K.size(); ++i)
	{
		cout << particle[1].K[i] << endl;
	}

	delete[] particle; //Free memory
	delete[] stream;

	//	return 0;
}


