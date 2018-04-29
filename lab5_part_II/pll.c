
int pll(int *NRZ, int nNRZ, int *bits, int *idx, int *pll, int *ppll, int dpll, float a){

	/* function to compute the nudge pll fast
	 *
	 * Inputs:
	 * 	NRZ  -  an array of the input Non-return-Zero signal
	 * 	nNRZ -  size of the array
	 * 	bits -  pre-allocated array to store the values of the sample bits
	 * 	idx  -  pre-allocated array to store the indexes for sampling the bits
	 * 	*pll -  pointer for the intial pll value variable
	 * 	*ppll - pointer for the previous pll value 
	 * 	dpll  - step size of the pl
	 * 	a     - nudge factor
	 *
	 * 	The function will store the resulting bit values in bits
	 * 	The function will store the indexes of the array for sampling in idx
	 *
	 *
	 * commandline to compile: cc -fPIC -shared -o libpll.so pll.c
	 */

	int i;
	int c = 0;


        
	if ((*pll < 0) && (*ppll > 0))
	{
		idx[c] = 0;
		bits[c] = NRZ[0];
		c++;
	}
	*ppll = *pll;
	*pll = *pll + dpll ;

	for (i=1 ; i < nNRZ ; i++) 
	{
		if  ((*pll < 0) && (*ppll > 0))
		{
			idx[c] = i;
			bits[c] = NRZ[i];
			c++;
		}
			
		if (NRZ[i-1] != NRZ[i])
			*pll = (int)((float)*pll * a);

		*ppll = *pll;
		*pll = *pll + dpll;
	}

	return c;
}

				
