/**************************************************************************
 *
 *   ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, Idx, PatNum, TransNum)
 *   prhs[0] - image size   
 *   prhs[1] - patch size
 *   prhs[2] - sliding distance
 *   prhs[3] - indices of image patches
 *   prhs[4] - the number of image patches
 *   prhs[5] - the number of square transforms
 *   plhs[0] - cluster map for every voxel
 *
 *   XuehangZheng, Zhipeng Li, UM-SJTU Joint Institute
 *
 *************************************************************************/


#include "mex.h"
#include <string.h>

/* Input Arguments */

#define	B_IN   prhs[0]
#define N_IN   prhs[1]
#define SZ_IN  prhs[2]
#define S_IN   prhs[3]
#define X_IN   prhs[4]
#define XS_IN  prhs[5]



/* Output Arguments */

#define	X_OUT	plhs[0]


void mexFunction(int nlhs, mxArray *plhs[], 
		         int nrhs, const mxArray*prhs[])
     
{ 
    float *x, *x_tmp, *x_lasttmp;
    double *s, *v;
    int numblock, patnum;
    mwSize sz[3], stepsize[3], n[3], ndims;
    mwIndex i, j, k, l, m, t, blocknum, a, num;
    mxArray *X_TMP, *X_lastTMP;
    
    
     /* Check for proper number of arguments */
    
    if (nrhs < 6 || nrhs > 7) 
    {
      mexErrMsgTxt("Invalid number of input arguments."); 
    }
    else if (nlhs > 1) 
    {
      mexErrMsgTxt("Too many output arguments."); 
    } 
    
    
    /* Check the input dimensions */ 
    
   
    if (!mxIsDouble(B_IN) ||  mxIsComplex(B_IN) || mxGetNumberOfDimensions(B_IN)>2) 
    {
      mexErrMsgTxt("Invalid input image size.");
    }

    ndims = mxGetM(B_IN)*mxGetN(B_IN);


    if (ndims<2 || ndims>3) 
    {
      mexErrMsgTxt("Output matrix can only be 2-D or 3-D.");
    }
    if (!mxIsDouble(N_IN) || mxIsComplex(N_IN) || mxGetNumberOfDimensions(N_IN)>2 || mxGetM(N_IN)*mxGetN(N_IN)!=ndims) {
      mexErrMsgTxt("Invalid patch size.");
    }
    if (nrhs == 6) 
    {
      if ( !mxIsDouble(SZ_IN) || mxIsComplex(SZ_IN) || mxGetNumberOfDimensions(SZ_IN)>2 || mxGetM(SZ_IN)*mxGetN(SZ_IN)!=ndims) 
      {
        mexErrMsgTxt("Invalid step size.");
      }
    }
     /* Check the the number of patchs */
     if (!mxIsDouble(X_IN) || mxIsComplex(X_IN) || mxGetM(X_IN)*mxGetN(X_IN)!=1)
    {
      mexErrMsgTxt("Invalid patch numbers.");
    }
    
     /* Check the number of square transforms */
     if (!mxIsDouble(XS_IN) || mxIsComplex(XS_IN) || mxGetM(XS_IN)*mxGetN(X_IN)!=1)
    {
      mexErrMsgTxt("Invalid transform numbers.");
    }
    
     /* Check the indices of image patches */
    if (!mxIsDouble(S_IN) ||  mxIsComplex(S_IN) || mxGetNumberOfDimensions(S_IN)>2) 
    {
      mexErrMsgTxt("Invalid indices of image patches.");
    }
    
    /* Get parameters */
    
    s = mxGetPr(B_IN);
    if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) 
    {
      mexErrMsgTxt("Invalid input image size.");
    }
    n[0] = (mwSize)(s[0] + 0.01);
    n[1] = (mwSize)(s[1] + 0.01);
    n[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    s = mxGetPr(N_IN);
    if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) 
    {
      mexErrMsgTxt("Invalid patch size.");
    }
    sz[0] = (mwSize)(s[0] + 0.01);
    sz[1] = (mwSize)(s[1] + 0.01);
    sz[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    if (nrhs == 6)
    {
      s = mxGetPr(SZ_IN);
      if (s[0]<1 || s[1]<1 || (ndims==3 && s[2]<1)) 
      {
        mexErrMsgTxt("Invalid step size.");
      }
      stepsize[0] = (mwSize)(s[0] + 0.01);
      stepsize[1] = (mwSize)(s[1] + 0.01);
      stepsize[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    }
    else
    {
      stepsize[0] = stepsize[1] = stepsize[2] = 1;
    }
    
    if (n[0]<sz[0] || n[1]<sz[1] || (ndims==3 && n[2]<sz[2])) 
    {
      mexErrMsgTxt("Patch size too large.");
    }
   
    /* Create a matrix for the return argument */
    
    X_OUT = mxCreateNumericArray(ndims, n, mxSINGLE_CLASS, mxREAL);
    X_TMP = mxCreateNumericArray(ndims, n, mxSINGLE_CLASS, mxREAL);
    X_lastTMP = mxCreateNumericArray(ndims, n, mxSINGLE_CLASS, mxREAL);
                       
    /* Assign pointers */
    
    x_tmp = (float *)mxGetPr(X_TMP);
    x_lasttmp = (float *)mxGetPr(X_lastTMP);
    x = (float *)mxGetPr(X_OUT);
    v = (double *)mxGetPr(S_IN);
            
   numblock = mxGetScalar(prhs[5]);
   patnum = mxGetScalar(prhs[4]);
      
    /* Do the actual computation and iterate over all blocks  */
    
 for(a=1;a<=numblock;a++)
{
      blocknum = 0;
    for (k=0; k<=n[2]-sz[2]; k+=stepsize[2]) 
     {
        for (j=0; j<=n[1]-sz[1]; j+=stepsize[1]) 
         {
           for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
            {
          
          /* add single block */
        
                if( (*(v+blocknum)==a) && (blocknum<patnum)  )
                  {
                       for (m=0; m<sz[2]; m++) 
                             {
                                for (l=0; l<sz[1]; l++) 
                                  {
                                      for (t=0; t<sz[0]; t++)
                                     {
                                           (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] +=1;
                                      }
                                   }
                              }
                   }
         
                  blocknum++;
          
            }
           if((i<n[0]-sz[0]+stepsize[0])&&(blocknum<patnum))
             {   
                    i=n[0]-sz[0];               
                        if((*(v+blocknum)==a)&&(blocknum<patnum))
                         {            
                            for (m=0; m<sz[2]; m++) 
                                {
                                   for (l=0; l<sz[1]; l++) 
                                    {
                                      for (t=0; t<sz[0]; t++)
                                      {
                                         (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] +=1;
                                       }
                                     }
                                  }
                          }
                
                      blocknum++;
               }
           }
       if((j<n[1]-sz[1]+stepsize[1])&&(blocknum<patnum))
          {   
              j=n[1]-sz[1];
                 for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
                   {          
                        /* add single block */
                      if((*(v+blocknum)==a)&&(blocknum<patnum))
                         {
                               for (m=0; m<sz[2]; m++) 
                                {
                                   for (l=0; l<sz[1]; l++) 
                                    {
                                      for (t=0; t<sz[0]; t++)
                                        {
                                          (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += 1;
                                         }
                                     }
                                  }
                          }
                       blocknum++;
          
                   }
                 if((i<n[0]-sz[0]+stepsize[0])&&(blocknum<patnum))
                   {   
                        i=n[0]-sz[0];
                           if((*(v+blocknum)==a)&&(blocknum<patnum))
                             {
                              for (m=0; m<sz[2]; m++) 
                               {
                                  for (l=0; l<sz[1]; l++) 
                                    {
                                      for (t=0; t<sz[0]; t++)
                                       {
                                         (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] +=1;
                                        }
                                     }
                                }
                              }
                         blocknum++;
                    }
          }
    }
   if((k<n[2]-sz[2]+stepsize[2])&&(blocknum<patnum))
          {   
              k=n[2]-sz[2];
                for (j=0; j<=n[1]-sz[1]; j+=stepsize[1]) 
                   {
                     for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
                       {         
                         /* add single block */
                           if((*(v+blocknum)==a)&&(blocknum<patnum))
                             {
                                for (m=0; m<sz[2]; m++) 
                                  {
                                     for (l=0; l<sz[1]; l++) 
                                       {
                                          for (t=0; t<sz[0]; t++)
                                            {
                                               (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += 1;
                                             }
                                        }
                                   }
                              }
                             blocknum++;
          
                         }
                      if((i<n[0]-sz[0]+stepsize[0])&&(blocknum<patnum))
                        {   
                           i=n[0]-sz[0];
                              if((*(v+blocknum)==a)&&(blocknum<patnum))
                               {
                                   for (m=0; m<sz[2]; m++) 
                                     {
                                       for (l=0; l<sz[1]; l++) 
                                         {
                                           for (t=0; t<sz[0]; t++)
                                            {
                                                (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += 1;
                                             }
                                          }
                                     }
                                 }
                           blocknum++;
                         }
                    }
                  if((j<n[1]-sz[1]+stepsize[1])&&(blocknum<patnum))
                    {   
                       j=n[1]-sz[1];
                          for (i=0; i<=n[0]-sz[0]; i+=stepsize[0]) 
                             {          
                                /* add single block */
                                if((*(v+blocknum)==a)&&(blocknum<patnum))
                                 { 
                                    for (m=0; m<sz[2]; m++) 
                                      {
                                         for (l=0; l<sz[1]; l++) 
                                          {
                                           for (t=0; t<sz[0]; t++)
                                            {
                                                (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] +=1;
                                             }
                                           }
                                       }
                                   }
                                blocknum++;
          
                             }
                          if((i<n[0]-sz[0]+stepsize[0])&&(blocknum<patnum))
                             {   
                                i=n[0]-sz[0];
                                   if((*(v+blocknum)==a)&&(blocknum<patnum))
                                      {
                                        for (m=0; m<sz[2]; m++) 
                                          {
                                            for (l=0; l<sz[1]; l++) 
                                              {
                                                for (t=0; t<sz[0]; t++)
                                                  {
                                                    (x_tmp+(k+m)*n[0]*n[1]+(j+l)*n[0]+i)[t] += 1;
                                                   }
                                               }
                                           }
                                       }
                                   blocknum++;
                               }
                  }
          }
 /*compare the index of different transforms  */   
            
       for(num=0;num<n[0]*n[1]*n[2];num++)
     {
                   if (*(x_tmp+num) >= *(x_lasttmp+num))
                   {
                       *(x+num)=a;
                       *(x_lasttmp+num)=*(x_tmp+num); 
                   }
                    
                   *(x_tmp+num)=0;
     }
     
}
    return;
}
