%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('../data/2Dxcat'));
addpath(genpath('../toolbox'));

%% setup target geometry and weight
down = 1; % downsample rate
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
ig = image_geom('nx', 420, 'dx', 500/512, 'down', down);
ig.mask = ig.circ > 0;
A = Gtomo2_dscmex(sg, ig, 'nthread', jf('ncore')*2);
% if neccessary, one could modify maxNumComThreads/jf('ncore') to make full
% use of threads of your machine to accelerate the computation of 
% forward and back projections. 

%% load external parameter
I0 = 1e4; % photon intensity

% load PWLS-EP Recon as initialization: xrlalm
load('xrlalm1e4_l2b16_os24_iter50.mat'); % change intial EP image when I0 is changed!
% load('xrlalm5e3_l2b16dot5_os24_iter50.mat');

%load transform: mOmega
load('slices_block15_iter1000_gamma125_31l0.mat');
mOmega = info.mOmega; clear info

%load ground truth image: xtrue
load('../data/2Dxcat/slice420.mat');

dir = ['../data/2Dxcat/tmp/' num2str(I0)];
printm('Loading external sinogram, weight, fbp...');
load([dir '/sino_fan.mat']);
load([dir '/wi.mat']);
load([dir '/xfbp.mat']);
% figure name 'xfbp'
% imshow(xfbp, [800 1200]);
% printm('Pre-calculating denominator D_A...');
% denom = A' * col(reshape(sum(A'), size(wi)) .* wi);
% denom=  A'*(wi(:).*(A*ones(size(xrla_msk,1),1)));
load([dir '/denom.mat']);

%% setup edge-preserving regularizer
ImgSiz =  [ig.nx ig.ny];  % image size
PatSiz =  [8 8];         % patch size
SldDist = [1 1];         % sliding distance

nblock = 4;            % Subset Number
nIter = 2;             % I--Inner Iteration
nOuterIter = 1000;     % T--Outer Iteration
CluInt = 1;            % Clustering Interval
isCluMap = 0;          % The flag of caculating cluster mapping
pixmax = inf;          % Set upper bond for pixel values

Ab = Gblock(A, nblock); clear A

% pre-compute D_R
numBlock = size(mOmega, 3);
vLambda = [];
for k = 1:size(mOmega, 3)
    vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
end
maxLambda = max(vLambda); clear vLambda;

PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
PatNum = size(PP, 2);
KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);

load([dir '/kappa.mat']);
KapPatch = im2colstep(kappa, PatSiz, SldDist); clear kappa;
KapPatch = mean(KapPatch,1);
%         KapPatch = repmat(KapPatch, prod(PatSiz), 1);
Kappa = col2imstep(single(repmat(KapPatch, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);


KapType = 1;

switch KapType
    
    case 0  % no patch-based weighting
        beta = 2e5;
        gamma = 20;
        
        D_R = 2 * beta * KK(ig.mask) * maxLambda; clear PP KK
        % D_R = 2 * beta * prod(PatSiz)/ prod(SldDist) * maxLambda;
        R = Reg_OST(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, mOmega, numBlock, CluInt);
        
    case 1  % patch-based weighting \tau * { \|~~\|_2 + \|~~\|_0 }
        beta = 1.3e4;
        gamma = 22;
        
        D_R = 2 * beta * Kappa(ig.mask) * maxLambda;  clear maxLambda Kappa;
        % construct regularizer R(x)
        R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, KapPatch, mOmega, numBlock, CluInt);
            
end
fprintf('KapType= %g, beta = %.1e, gamma = %g: \n\n', KapType, beta, gamma);


info = struct('intensity',I0,'ImgSiz',ImgSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
    'nblock',nblock,'nIter',nIter,'CluInt',CluInt,'pixmax',pixmax,'transform',mOmega,...
    'xrla',[],'vIdx',[],'ClusterMap',[], 'RMSE',[],'SSIM',[],'relE',[],'perc',[],'idx_change_perc',[],'cost',[]);

xini = xrlalm .* ig.mask;    %initial EP image
xrla_msk = xrlalm(ig.mask);
% xini = xfbp .* ig.mask;     %initial FBP image
% xrla_msk = xfbp(ig.mask);   clear xfbp
info.xrla = xini;

%% Recon
SqrtPixNum = sqrt(sum(ig.mask(:)>0)); % sqrt(pixel numbers in the mask)
stop_diff_tol = 1e-3; % HU
iterate_fig = figure(55);
idx_old = ones([1,PatNum],'single');

for ii=1:nOuterIter
    %     figure(iterate_fig); drawnow;
    xold = xrla_msk;
    AAA(1,ii) = norm(xrla_msk - xtrue(ig.mask)) / SqrtPixNum;
    fprintf('RMSE = %g\n', AAA(1,ii));
    info.RMSE = AAA(1,:);
    AAA(2,ii)= ssim(info.xrla, xtrue);
    fprintf('SSIM = %g\n', AAA(2,ii));
    info.SSIM = AAA(2,:);
    
    fprintf('Iteration = %d:\n', ii);
    [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(sino, '2d'),  reshaper(wi, '2d'),...
        R, denom, D_R, 'pixmax', pixmax, 'chat', 0, 'alpha', 1.999, 'rho', [],'niter', nIter);
    
    [info.perc(:,ii),info.vIdx] = R.nextOuterIter();
    fprintf('perc = %g\n', info.perc(:,ii));
    
    info.idx_change_perc(:,ii) = nnz(idx_old - info.vIdx)/PatNum;
    fprintf('Idx Change Perc = %g\n', info.idx_change_perc(:,ii));
    idx_old = info.vIdx;
    
    %     info.cost(:,ii) = cost;
    info.relE(:,ii) =  norm(xrla_msk - xold) / SqrtPixNum;
    fprintf('relE = %g\n', info.relE(:,ii));
    if info.relE(:,ii) < stop_diff_tol
        break
    end
    
    info.xrla = ig.embed(xrla_msk);
    figure(120), imshow(info.xrla, [800 1200]); drawnow;
    %     info.ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numBlock);
    %     figure(iterate_fig); drawnow;
    %     subplot(2,3,1),imshow((info.ClusterMap == 1) .* info.xrla, [800,1200]);
    %     subplot(2,3,2);imshow((info.ClusterMap == 2) .* info.xrla, [800,1200]);
    %     subplot(2,3,3);imshow((info.ClusterMap == 3) .* info.xrla, [800,1200]);
    %     subplot(2,3,4);imshow((info.ClusterMap == 4) .* info.xrla, [800,1200]);
    %     subplot(2,3,5);imshow((info.ClusterMap == 5) .* info.xrla, [800,1200]);
    
end

%% Calculate clusterMap
% if(isCluMap == 1)
%    info.ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numBlock);
% %    figure(55), imshow(info.ClusterMap, [1,numBlock]);colorbar
%    figure(55), imshow((info.ClusterMap == numBlock) .* info.xrla, [800,1200]);colorbar
% end

%%
% save(sprintf('./result/%.1ekap1_blk%g_beta%.1e_gam%g_clu%g_learn125.mat', I0, ...
%     numBlock, beta, gamma, CluInt), 'info')
% imshow(cat(2, info.xrla, xini), [800 1200]);


% figure name 'SSIM'
% plot(info.SSIM);
% xlabel('Number of Total Iteration','fontsize',18)
% ylabel('SSIM','fontsize',18)
% legend('PWLS-ST')

figure name 'RMSE'
plot(info.RMSE,'-+')
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
legend('PWLS-ULTRA')

figure name 'compare'
imshow(cat(2, info.xrla, xini), [800 1200]);

% save('info.mat', 'info')

