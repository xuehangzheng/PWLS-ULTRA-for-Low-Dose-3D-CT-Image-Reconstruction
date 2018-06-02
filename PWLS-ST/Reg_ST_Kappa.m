classdef Reg_ST_Kappa < handle

 properties
    mMask;    % the mask matrix
    PatSiz;   % patch size
    ImgSiz;   % image size
    SldDist;  % sliding distance
    beta;
    gamma;    % threshold
    mOmega;   % transform matrix  
    KapPatch
    mSpa;     % the matrix of sparse code
    rstSpa;   % the flag of sparsecode updating       
 end
    
 methods
   function obj = Reg_ST_Kappa(mask, PatSiz, ImgSiz, SldDist, beta, gamma, KapPatch, mOmega)
     obj.mMask = mask;
     obj.PatSiz = PatSiz; 
     obj.ImgSiz = ImgSiz;
     obj.SldDist = SldDist;
     obj.beta = beta;
     obj.gamma = gamma;
     obj.KapPatch = KapPatch;
     obj.mOmega = mOmega;
     obj.rstSpa = true;
   end
        
   function cost = penal(obj, A, x, wi, sino)
     % data fidelity     
      df = .5 * sum(col(wi) .* (A * x - col(sino)).^2, 'double');
      fprintf('df = %g\n', df);      
       x = embed(x, obj.mMask);  
       mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
       mCod = obj.mOmega * mPat;  clear mPat;        
      % sparsity error
       spa_err = obj.beta * sum( obj.KapPatch(1,:).* sum((mCod - obj.mSpa).^2,1) ); 
       clear mCod;
       fprintf('se = %g\n', spa_err);
       spa = obj.beta * obj.gamma^2 * nnz(obj.mSpa);% l0norm
       fprintf('sp = %g\n', spa); 
       cost_val = df + spa_err + spa;
       cost=[]; cost(1)= cost_val; cost(2)= df; 
       cost(3)= spa_err; cost(4)= spa; 
   end
   
   function grad = cgrad(obj, x)
     x = embed(x, obj.mMask);
     mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
     % update sparse code only at the first inner iteration
     if(obj.rstSpa)   
       mCod = obj.mOmega * mPat; clear mPat;
       % hard-thresholding
%        obj.mSpa= mCod.*(bsxfun(@ge,abs(mCod),obj.gamma./sqrt(obj.KapPatch))); 
       obj.mSpa = mCod .* (abs(mCod) > (obj.gamma./sqrt(obj.KapPatch)));
       diff = obj.mOmega' * (mCod - obj.mSpa); clear mCod;       
       obj.rstSpa = false;
     else
       diff = obj.mOmega' * (obj.mOmega * mPat - obj.mSpa); clear mPat; 
     end
     diff = obj.KapPatch .* diff;  
     grad = 2 * obj.beta .* col2imstep(single(diff), obj.ImgSiz, obj.PatSiz, obj.SldDist); 
     grad = grad(obj.mMask); 
   end
     
            
   function perc = nextOuterIter(obj)
    % set the flag of updating SparseCode
    obj.rstSpa = true;
    % sparsity check
    perc = nnz(obj.mSpa) / numel(obj.mSpa) * 100; 
   end   
   
  end
    
end

