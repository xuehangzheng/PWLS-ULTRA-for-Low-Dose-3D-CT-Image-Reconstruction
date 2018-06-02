classdef Reg_OST_Kappa < handle
    
    properties
        mMask;    % the mask matrix
        ImgSiz;   % image size
        PatSiz;   % patch size
        SldDist;  % sliding distance
        beta;
        gamma;    % threshold
        KapPatch;
        mOmega;   % transform matrix
        numBlock; % the number of square transforms
        vIdx;     % the patch index decides which transform belongs to
        isSpa;    % the flag of sparse code update
        isClu;    % the flag of clustering update
        CluInt;   % the number of clustering interval
        mSpa;     % the matrix of sparse code
    end
    
    methods
        function obj = Reg_OST_Kappa(mask, ImgSiz, PatSiz, SldDist, beta, gamma, KapPatch, mOmega, numBlock, CluInt)
            obj.mMask = mask;
            obj.PatSiz = PatSiz;
            obj.ImgSiz = ImgSiz;
            obj.SldDist = SldDist;
            obj.beta = beta;
            obj.gamma = gamma;
            obj.mOmega = mOmega;
            obj.KapPatch = KapPatch;
            obj.numBlock = numBlock;
            obj.isSpa = true;
            obj.isClu = CluInt;
            obj.CluInt = CluInt;
            obj.vIdx = [];
        end
        
        function cost = penal(obj, A, x, wi, sino)
            % data fidelity
            df = .5 * sum(col(wi) .* (A * x - col(sino)).^2, 'double');
            fprintf('df = %g\n', df);
            x = embed(x, obj.mMask);
            mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
            mCod = zeros(size(mPat), 'single');
            for k = 1 : obj.numBlock
                tmp = obj.vIdx==k;
                mCod(:,tmp) = obj.mOmega(:,:,k) * mPat(:,tmp) ;
            end
            clear mPat
            % sparsity error
            spa_err = obj.beta * sum( obj.KapPatch.*sum((mCod - obj.mSpa).^2,1) ); clear mCod;
            fprintf('se = %g\n', spa_err);
            spa = obj.beta * obj.gamma^2 * sum(obj.KapPatch .* sum(obj.mSpa~=0,1));
            fprintf('sp = %g\n', spa);
            
            cost_val = df + spa_err + spa;
            cost=[]; cost(1)= cost_val; cost(2)= df;
            cost(3)= spa_err; cost(4)= spa;
        end
        
        function grad = cgrad(obj, x)
            x = embed(x, obj.mMask);
            mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist);  clear x;
            %%%%%%%%%%%%% cluster index update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(obj.isClu == obj.CluInt)
                numPatch = size(mPat, 2);
                error = zeros(obj.numBlock, numPatch, 'double');
                %%%%%%%%%%%%%% clustering measure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for k = 1 : obj.numBlock
                    a1 = obj.mOmega(:,:,k) * mPat;
                    % hard-thresholding
                    a0 = a1 .* (abs(a1) >= obj.gamma);
                    error(k,:) = sum((a1-a0).^2,'double') + obj.gamma^2 * sum(abs(a0)>0);
                end
                clear a0 a1;
                %%%%%%%%%%%%%% clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if(isempty(obj.vIdx))
                    obj.vIdx = ones(1, numPatch);
                end
                [~, obj.vIdx] = min(error, [] ,1);  clear error;
                
                obj.isClu = 0; % reset clustering counter
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            diff = zeros(size(mPat), 'single');
            if(obj.isSpa)
                mCod = zeros(size(mPat), 'single');
                for k = 1 : obj.numBlock
                    tmp = obj.vIdx==k;
                    mCod(:,tmp) = obj.mOmega(:,:,k) * mPat(:,tmp) ;
                end
                clear mPat
                % hard-thresholding
                %          obj.mSpa = mCod .* (abs(mCod) >= (obj.gamma./sqrt(obj.KapPatch)));
                obj.mSpa = mCod .* (abs(mCod) >= obj.gamma);
                
                for k = 1:obj.numBlock
                    tmp = obj.vIdx==k;
                    diff(:,tmp) = obj.mOmega(:,:,k)' * (mCod(:,tmp) - obj.mSpa(:,tmp));
                end
                clear mCod
                obj.isSpa = false;  % close the flag of sparse code update
            else
                for k = 1:obj.numBlock
                    tmp = obj.vIdx==k;
                    diff(:,tmp) = obj.mOmega(:,:,k)' * ( obj.mOmega(:,:,k) * mPat(:,tmp) - obj.mSpa(:,tmp) );
                end
                clear mPat
            end
            diff = obj.KapPatch .* diff;
            grad = 2 * obj.beta .* col2imstep(single(diff), obj.ImgSiz, obj.PatSiz, obj.SldDist);
            grad = grad(obj.mMask);
        end
        
        
        function [perc,vIdx] = nextOuterIter(obj)
            obj.isClu = obj.isClu + 1;
            vIdx = obj.vIdx;
            obj.isSpa = true; % open the flag of updating sparse code
            % sparsity check
            perc = nnz(obj.mSpa) / numel(obj.mSpa) * 100;
        end
    end
    
end

