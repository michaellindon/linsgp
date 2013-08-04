#include <cstdlib>
#include <iostream>
#include <vector>
#include <new>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
#include <omp.h>
#include <trng/normal_dist.hpp>

class particles{
	public:
		double phi,phit,Z;
};

using namespace std;

int main(int argc, char *argv[])
{
	int i;
	int np; //np ~ number of particles
	int niter; //niter ~ number of iterations
	int nthreads; //nthreads ~ number of threads
	int rank; //thread identifier

	particles * particle; 
	trng::normal_dist<> normal(0.0,1.0);
	np=atoi(argv[1]); //Get np from user
	particle = new (nothrow)  particles[np]; //Create Particles

	//Check if memory allocation was successful
	if (particle==0){
		std::cout << "Dynamic memory allocation for " 
			<< np
			<< " particles was not successful." 
			<< std::endl;
		std::cout << "Please consider using fewer particles." 
			<< std::endl;
		return EXIT_FAILURE;
	}


	//Query maximum threads supported by hardware
	nthreads=omp_get_max_threads();
	//Set number of threads to the supported maximum
	omp_set_num_threads(nthreads);
	//Create "nthreads" number of independent random number streams 
	//trng::yarn2 stream[nthreads];
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



#pragma omp parallel for
	for (i = 0; i < np; ++i)
	{
		rank=omp_get_thread_num();
		particle[i].phi = normal(stream[rank]);
	}


	cout << particle[3].phi << endl;
	cout << particle[4].phi << endl;
	cout << particle[5].phi << endl;
	delete[] particle; //Free memory

	return 0;
}
