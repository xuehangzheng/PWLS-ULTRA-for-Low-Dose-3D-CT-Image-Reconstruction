classdef Reg_DL < handle
    
    properties
        mMask;  % the mask matrix
        PatSiz; % [rows cols thickness] 1x3 vector patch size
        ImgSiz; % [rows cols thickness] 1x3 vector
        SldDist; % sliding distance
        beta;
        maxatoms;      % maximal number of atoms
        EPSILON; % sparsification error
        mOmega; % transform matrix
        mSpa;   % the matrix of sparse code
        rstSpa; % the flag of sparsecode updating
        param;
    end
    
    methods
        function obj = Reg_DL(mask, PatSiz, ImgSiz, SldDist, beta, maxatoms, EPSILON, mOmega)
            obj.mMask = mask;
            obj.PatSiz = PatSiz;
            obj.ImgSiz = ImgSiz;
            obj.SldDist = SldDist;
            obj.beta = beta;
            obj.maxatoms = maxatoms;
            obj.EPSILON = EPSILON;
            obj.mOmega = mOmega;
            obj.rstSpa = true;
        end
        
        function cost = penal(obj, A, x, wi, sino)
            % data fidelity
            df = .5 * sum(col(wi) .* (A * x - col(sino)).^2, 'double');
            fprintf('df = %g\n', df);
            x = embed(x, obj.mMask);
            mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
            % sparsity error
            spa_err = obj.beta *  sum(col(mPat - obj.mOmega * obj.mSpa).^2); clear mCod;
            fprintf('se = %g\n', spa_err);
            spa = nnz(obj.mSpa);
            fprintf('sp = %g\n', spa);
            cost=[]; cost(1)= df; cost(2)= spa_err; cost(3)= spa;
        end
        
        function grad = cgrad(obj, x)
            x = embed(x, obj.mMask);
            mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
            % update sparse code only at the first inner iteration
            if(obj.rstSpa)
                G = obj.mOmega'* obj.mOmega;
                obj.mSpa = omp2(double(obj.mOmega' * mPat), double(sum(mPat.*mPat)), ...
                    G, obj.EPSILON, 'gammamode','full','maxatoms', obj.maxatoms);
                diff = mPat - obj.mOmega * obj.mSpa;
                obj.rstSpa = false;
            else
                diff = mPat - obj.mOmega * obj.mSpa;
            end
            clear mPat;
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

