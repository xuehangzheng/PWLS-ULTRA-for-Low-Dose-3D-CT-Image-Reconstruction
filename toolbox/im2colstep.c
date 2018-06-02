/**************************************************************************
 *
 * File name: im2colstep.c
 *
 * Ron Rubinstein
 * Computer Science Department
 * Technion, Haifa 32000 Israel
 * ronrubin@cs
 *
 * Last Updated: 31.8.2009
 *
 * Modified into single precision
 * by Xuehang Zheng, UM-SJTU Joint Institute
 *************************************************************************/


#include "mex.h"
#include <string.h>


/* Input Arguments */

#define	X_IN	 prhs[0]
#define SZ_IN  prhs[1]
#define S_IN   prhs[2]


/* Output Arguments */

#define	B_OUT	plhs[0]


void mexFunction(int nlhs, mxArray *plhs[], 
		             int nrhs, const mxArray*prhs[])
     
{ 
    float *x, *b;
    double *s;
    mwSize sz[3], stepsize[3], n[3], ndims, pdim[2];
    mwIndex i, j, k, l, m, blocknum;
    
    
    /* Check for proper number of arguments */
    
    if (nrhs < 2 || nrhs > 3) {
      mexErrMsgTxt("Invalid number of input arguments."); 
    } else if (nlhs > 1) {
      mexErrMsgTxt("Too many output arguments."); 
    } 
    
    
    /* Check the the input dimensions */ 
    
    ndims = mxGetNumberOfDimensions(X_IN);
    
    if (!mxIsSingle(X_IN) || mxIsComplex(X_IN) || ndims>3) {
      mexErrMsgTxt("X should be a 2-D or 3-D single matrix.");
    }
    if (!mxIsDouble(SZ_IN) || mxIsComplex(SZ_IN) || mxGetNumberOfDimensions(SZ_IN)>2 || mxGetM(SZ_IN)*mxGetN(SZ_IN)!=ndims) {
      mexErrMsgTxt("Invalid block size.");
    }
    if (nrhs == 3) {
      if ( !mxIsDouble(S_IN) || mxIsComplex(S_IN) || mxGetNumberOfDimensions(S_IN)>2 || mxGetM(S_IN)*mxGetN(S_IN)!=ndims) {
        mexErrMsgTxt("Invalid step size.");
      }
    }
    
    
    /* Get parameters */

    s = mxGetPr(SZ_IN);
    if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) {
      mexErrMsgTxt("Invalid block size.");
    }
    sz[0] = (mwSize)(s[0] + 0.01);
    sz[1] = (mwSize)(s[1] + 0.01);
    sz[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    if (nrhs == 3) {
      s = mxGetPr(S_IN);
      if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) {
        mexErrMsgTxt("Invalid step size.");
      }
      stepsize[0] = (mwSize)(s[0] + 0.01);
      stepsize[1] = (mwSize)(s[1] + 0.01);
      stepsize[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    }
    else {
      stepsize[0] = stepsize[1] = stepsize[2] = 1;
    }
    
     n[0] = (mxGetDimensions(X_IN))[0];
     n[1] = (mxGetDimensions(X_IN))[1];
     n[2] = ndims==3 ? (mxGetDimensions(X_IN))[2] : 1;

    if (n[0]<sz[0] || n[1]<sz[1] || (ndims==3 && n[2]<sz[2])) {
      mexErrMsgTxt("Block size too large.");
    }
    

    pdim[0] = sz[0]*sz[1]*sz[2];
    
    /* Create a matrix for the return argument */
     if( (n[0]-sz[0])%stepsize[0]==0 && (n[1]-sz[1])%stepsize[1]==0 && (n[2]-sz[2])%stepsize[2]==0 )
    {
       pdim[1] =  ((n[0]-sz[0])/stepsize[0]+1)*((n[1]-sz[1])/stepsize[1]+1)*((n[2]-sz[2])/stepsize[2]+1);
     }
     else if( (n[0]-sz[0])%stepsize[0]!=0 && (n[1]-sz[1])%stepsize[1]==0 && (n[2]-sz[2])%stepsize[2]==0 )
    {
       pdim[1] = ((stepsize[0]-(n[0]-sz[0])%stepsize[0]+n[0]-sz[0])/stepsize[0]+1)*((n[1]-sz[1])/stepsize[1]+1)*((n[2]-sz[2])/stepsize[2]+1);
     }
     else  if( (n[1]-sz[1])%stepsize[1]!=0 && (n[0]-sz[0])%stepsize[0]==0 && (n[2]-sz[2])%stepsize[2]==0)
    {
       pdim[1] = ((n[0]-sz[0])/stepsize[0]+1)*((stepsize[1]-(n[1]-sz[1])%stepsize[1]+n[1]-sz[1])/stepsize[1]+1)*((n[2]-sz[2])/stepsize[2]+1);
     }
     else if( (n[2]-sz[2])%stepsize[2]!=0 && (n[0]-sz[0])%stepsize[0]==0 && (n[1]-sz[1])%stepsize[1]==0 )
    {
      pdim[1] = ((n[0]-sz[0])/stepsize[0]+1)*((n[1]-sz[1])/stepsize[1]+1)*((stepsize[2]-(n[2]-sz[2])%stepsize[2]+n[2]-sz[2])/stepsize[2]+1);
     }
     else if( (n[2]-sz[2])%stepsize[2]==0 && (n[0]-sz[0])%stepsize[0]!=0 && (n[1]-sz[1])%stepsize[1]!=0 )
    {
      pdim[1] = ((stepsize[0]-(n[0]-sz[0])%stepsize[0]+n[0]-sz[0])/stepsize[0]+1)*((stepsize[1]-(n[1]-sz[1])%stepsize[1]+n[1]-sz[1])/stepsize[1]+1)*((n[2]-sz[2])/stepsize[2]+1);
     }
    else if( (n[2]-sz[2])%stepsize[2]!=0 && (n[0]-sz[0])%stepsize[0]!=0 && (n[1]-sz[1])%stepsize[1]==0 )
    {
      pdim[1] = ((stepsize[0]-(n[0]-sz[0])%stepsize[0]+n[0]-sz[0])/stepsize[0]+1)*((n[1]-sz[1])/stepsize[1]+1)*((stepsize[2]-(n[2]-sz[2])%stepsize[2]+n[2]-sz[2])/stepsize[2]+1);
     }
      else if( (n[2]-sz[2])%stepsize[2]!=0 && (n[0]-sz[0])%stepsize[0]==0 && (n[1]-sz[1])%stepsize[1]!=0 )
    {
      pdim[1] = ((n[0]-sz[0])/stepsize[0]+1)*((stepsize[1]-(n[1]-sz[1])%stepsize[1]+n[1]-sz[1])/stepsize[1]+1)*((stepsize[2]-(n[2]-sz[2])%stepsize[2]+n[2]-sz[2])/stepsize[2]+1); 
     }
      else 
    {
      pdim[1] = ((stepsize[0]-(n[0]-sz[0])%stepsize[0]+n[0]-sz[0])/stepsize[0]+1)*((stepsize[1]-(n[1]-sz[1])%stepsize[1]+n[1]-sz[1])/stepsize[1]+1)*((stepsize[2]-(n[2]-sz[2])%stepsize[2]+n[2]-sz[2])/stepsize[2]+1);   
    }

    B_OUT = mxCreateNumericArray(2, pdim, mxSINGLE_CLASS, mxREAL);
  
    /* Assign pointers */

    x = (float *)mxGetPr(X_IN);
    b = (float *)mxGetPr(B_OUT);
            
    
    /* Do the actual computation */
    
    blocknum = 0;
    
    /* iterate over all blocks */
 
  for (k=0; k<=n[2]-sz[2]; k+=stepsize[2]) 
    {
      for (j=0; j<=n[1]-sz[1]; j+=stepsize[1]) 
      {
         
        for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
 
          /* copy single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
            }
          }
          blocknum++;
        }
           if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
               for (m=0; m<sz[2]; m++) 
               {
                  for (l=0; l<sz[1]; l++) 
                   {
                   memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
                    }
                }
          blocknum++;
                        
              
           }
      }
       if(j<n[1]-sz[1]+stepsize[1])
      {
              j=n[1]-sz[1];
               for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
 
          /* copy single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
            }
          }
          blocknum++;
        }
           if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
               for (m=0; m<sz[2]; m++) 
               {
                  for (l=0; l<sz[1]; l++) 
                   {
                   memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
                    }
                }
          blocknum++;
                        
              
           }
       }
    }
    if(k<n[2]-sz[2]+stepsize[2])
    {
        k=n[2]-sz[2];
        for (j=0; j<=n[1]-sz[1]; j+=stepsize[1]) 
      {
         
        for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
 
          /* copy single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
            }
          }
          blocknum++;
        }
           if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
               for (m=0; m<sz[2]; m++) 
               {
                  for (l=0; l<sz[1]; l++) 
                   {
                   memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
                    }
                }
          blocknum++;
                        
              
           }
      }
       if(j<n[1]-sz[1]+stepsize[1])
      {
              j=n[1]-sz[1];
               for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
        {
 
          /* copy single block */
          for (m=0; m<sz[2]; m++) 
          {
            for (l=0; l<sz[1]; l++) 
            {
              memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
            }
          }
          blocknum++;
        }
           if(i<n[0]-sz[0]+stepsize[0])
          {   
              i=n[0]-sz[0];
               for (m=0; m<sz[2]; m++) 
               {
                  for (l=0; l<sz[1]; l++) 
                   {
                   memcpy(b + blocknum*sz[0]*sz[1]*sz[2] + m*sz[0]*sz[1] + l*sz[0], x+(k+m)*n[0]*n[1]+(j+l)*n[0]+i, sz[0]*sizeof(float));
                    }
                }
          blocknum++;
                        
              
           }
       }
    }
    
    return;
}
