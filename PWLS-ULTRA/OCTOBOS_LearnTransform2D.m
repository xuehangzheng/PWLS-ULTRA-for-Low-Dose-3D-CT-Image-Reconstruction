%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('../toolbox'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('../data/3Dxcat/phantom154.mat');

PatSiz = 8 * [1 1];   % patch size
SldDist = 1 * [1 1];  % sliding distance
lambada0 = 31;        % set it large enough to guarantee well-conditioned
iter = 1000;          % iteration
gamma = 110;
numBlock = 5;         % number of clusters

patch = [];
for ii=[10 30 40 50 80]  % training data
    %   for ii = 45          % testing data
    image = downsample2(phantom(:, : , ii), 2);
    patch_tmp = im2colstep(single(image), PatSiz, SldDist);
    patch = [patch patch_tmp];
end
patch = double(patch); % "single" is not accurate enough for OCTOBOS training
ImgSiz = size(image);
PatNum = size(patch, 2);
fprintf('Length of training set: %d\n', PatNum);

[IDX, ~] = kmeans(patch',numBlock);   % K-mean Initialization
% IDX = randi(numBlock,PatNum,1);     % Random Initialization

TransWidth = prod(PatSiz);

D = kron(dctmtx(PatSiz(1)),dctmtx(PatSiz(2))); % DCT Initialization
mOmega = zeros(TransWidth,TransWidth,numBlock, 'double');
for i = 1:numBlock
    mOmega(:,:,i) = D;
end

perc = zeros(iter,numBlock,'single'); % sparsity (percentage)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1 : iter
    fprintf('iteration = %d:\n', j);
    for k = 1 : numBlock
        % sparse coding
        patch_k = patch(:, IDX == k); % this way is faster
        lambada_k = lambada0 * norm(patch_k, 'fro') ^ 2;% lambda_{k} update
        sparseCode = mOmega(:,:,k) * patch_k;
        % hard-thresholding
        sparseCode = sparseCode .*(abs(sparseCode) > gamma);
        perc(j,k) = nnz(sparseCode)/ numel(sparseCode)* 100;
        %         fprintf('sparsity(%d) = %g\n', k, perc(j,k));
        
        % transform update
        if (size(patch_k,2) > 0) % if patch_k is empty, transform will be unchange
            [U,S,V]=svd((patch_k*patch_k') + (lambada_k*eye(TransWidth)));
            LL2=(inv(U*(S^(1/2))*V')); clear U S V
            [Q1,Si,R]=svd(LL2*patch_k*sparseCode');
            sig=diag(Si); clear Si
            gamm=(1/2)*(sig + (sqrt((sig.^2) + 2*lambada_k)));
            B=R*(diag(gamm))*Q1';
            mOmega(:,:,k)=B*(LL2);
        else
            fprintf('patch %g is empty\n', k);
        end
    end
    clear patch_k sparseCode LL2 Q1 sig R B
    fprintf('Cond Number(%d) = %g\n', numBlock, cond(mOmega(:,:,numBlock)));
    fprintf('sparsity(%d) = %g\n', numBlock, perc(j,numBlock));
    error = zeros(numBlock, PatNum, 'double');
    %%%%%%%%% clustering measure %%%%%%%
    for k = 1 : numBlock
        a1 = mOmega(:,:,k) * patch;
        a0 = a1 .*(abs(a1) > gamma);
        error(k, :) = sum((a1-a0).^2,'double') + gamma^2 * sum(abs(a0) > 0) ...
            + lambada0 * sum(patch.^2,'double') *...
            (sum(col(mOmega(:,:,k).^2),'double') - log(abs(det(mOmega(:,:,k))))) ;
    end
    %%%%%%%%% clustering %%%%%%%%%%%%%%
    [~, IDX] = min(error, [] ,1);
    clear  error a1 a0
    
    %%%% check cluster-mapping %%%%%%%%
    CluMap = ClusterMap(ImgSiz, PatSiz, SldDist, IDX(:,0.8*end+1:end), PatNum, numBlock);
    figure(55); drawnow;
    % imshow(CluMap, [1,numBlock]);colorbar
    subplot(1,5,1); imshow((CluMap == 1).* image, [800,1200]);
    subplot(1,5,2); imshow((CluMap == 2).* image, [800,1200]);
    subplot(1,5,3); imshow((CluMap == 3).* image, [800,1200]);
    subplot(1,5,4); imshow((CluMap == 4).* image, [800,1200]);
    subplot(1,5,5); imshow((CluMap == 5).* image, [800,1200]);
    % IDX(:,0.8*end+1:end) corresponds to the last training slice used for visualization here.
end

%%%%%%%%%%%%%%%%%%%%%% check convergency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
for k = 1 : numBlock
    handles(k) = plot(perc(:,k));hold on;
    lables{k} = sprintf('cluster %d',k);
end
legend(handles,lables{:});
xlabel('Number of Iteration','fontsize',18)
ylabel('Sparity ( % )','fontsize',18)
%%%%%%%%%%%%%%%%%%%%%% check condition number %%%%%%%%%%%%%%%%%%%%%%%%%%%%
condTransform = zeros(numBlock, 1);
for k = 1 : numBlock
    condTransform(k, 1) = cond(mOmega(:, :, k));
end
%%%%%%%%%%%%%%%%%%%%% visualize learned transforms %%%%%%%%%%%%%%%%%%%%%%%
% info = struct('lambada0',lambada0,'ImgSiz',size(image),'SldDist',SldDist,'gamma',gamma,...
%  'numBlock',numBlock,'iter',iter,'mOmega',mOmega,'IDX',IDX ,'perc',perc);

transform = mOmega;
Taa=[]; Taaa=[];
for k = 1:size(transform, 3)
    for i=1:size(transform,1)
        transform(i,:,k) = transform(i,:,k)-min(transform(i,:,k));
        if(max(transform(i,:,k))>0)
            transform(i,:,k)=transform(i,:,k)/(max(transform(i,:,k)));
        end
    end
    
    jy=2;cc=1;
    Ta=(max(max(transform(:, :, k))))*ones((8+jy)*7 + 8,(8+jy)*7 + 8);
    for i=1:8+jy:(7*(8+jy))+1
        for j=1:8+jy:(7*(8+jy))+1
            Ta(i:i+7,j:j+7)=reshape((transform(cc,:,k))',8,8);
            cc=cc+1;
        end
    end
    blank = zeros(size(Ta,1),2);
    Taa = cat(2 , Ta , blank);
    Taaa = cat(2 , Taaa , Taa);
end
figure();imagesc(Taaa);colormap('Gray');axis off;axis image;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%