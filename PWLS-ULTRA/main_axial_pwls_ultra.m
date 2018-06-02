%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('../data/3Dxcat'));
addpath(genpath('../toolbox'));

%% setup target geometry and weight
down = 1; % downsample rate
cg = ct_geom('ge2', 'down', down);
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
ig.mask = ig.circ > 0;
A = Gcone(cg, ig, 'type', 'sf2','nthread', jf('ncore')*2);
% if neccessary, one could modify maxNumComThreads/jf('ncore') to make full
% use of threads of your machine to accelerate the computation of 
% forward and back projections. 

%% load external data
I0 = 1e4; % photon intensity

% load PWLS-EP Recon as initialization: xrlalm
load('1e4_l2b14dot5_os24_iter50.mat'); % change intial EP image when I0 is changed!
% load('5e3_l2b14dot5_os24_iter50.mat');

% load transform: mOmega
load('slice101_154_block15_SldDist2_iter1000_gamma75.mat');

% load ground truth image: xtrue
load('xtrue_crop17-80.mat');
mOmega = info.mOmega; clear info

% load measurements and initial data
dir = ['../data/3Dxcat/tmp/' num2str(I0)];
printm('Loading sinogram, weighting, xfdk...');
load([dir '/sino_cone.mat']);
load([dir '/wi.mat']);
% load([dir '/xfdk.mat']);
% figure; imshow(xfdk(:,:,end/2), [800 1200]);

%% 
xrla_msk = xrlalm(ig.mask);  % initial EP image
%xrla = xfdk .* ig.mask;     % initial FDK image
%xrla_msk = xfdk(ig.mask);

% set up ROI
start_slice = 17; end_slice = 80;
xroi = xrlalm(:,:,start_slice:end_slice); clear xrlalm
mask_roi = ig.mask(:,:,start_slice:end_slice);
% roi = ig.mask; roi(:,:,1:start_slice-1) = 0; roi(:,:,end_slice+1:end) = 0;
% roi = roi(ig.mask);

ImgSiz = [ig.nx ig.ny ig.nz]; % image size
PatSiz = [8 8 8];             % patch size
SldDist = 2 * [1 1 1];        % sliding distance

nblock = 4;            % Subset Number
nOuterIter = 1000;     % T--Outer Iteration
nIter = 2;             % I--Inner Iteration
CluInt = 20;           % Clustering Interval
isCluMap = 1;          % The flag of caculating cluster mapping
pixmax = inf;          % Set upper bond for pixel values

printm('Pre-calculating denominator: D_A...');
% denom = A' * col(reshape(sum(A'), size(wi)) .* wi);
% denom=  A'*(wi(:).*(A*ones(size(xrla_msk,1),1)));
load([dir '/denom.mat']);

Ab = Gblock(A, nblock); clear A

% pre-compute D_R
numBlock = size(mOmega, 3);
vLambda = [];
for k = 1:numBlock
    vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
end
maxLambda = max(vLambda); clear vLambda;


KapType = 0;

switch KapType
    case 0 
        beta= 2.5e5;
        gamma = 18;
        
        PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
        PatNum = size(PP, 2);
        KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);
        D_R = 2 * beta * KK(ig.mask) * maxLambda; clear PP KK
        % D_R = 2 * beta * prod(PatSiz)/ prod(SldDist) * maxLambda;
        R = Reg_OST(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, mOmega, numBlock, CluInt);
        
    case 1    
        beta= 1.5e4;
        gamma = 20;
        
        load([dir '/kappa.mat']);
        KapPatch = im2colstep(kappa, PatSiz, SldDist); clear kappa;
        PatNum = size(KapPatch, 2);
        KapPatch = mean(KapPatch,1);
%         KapPatch = repmat(KapPatch, prod(PatSiz), 1);
        Kappa = col2imstep(single(repmat(KapPatch, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);
        D_R = 2 * beta * Kappa(ig.mask) * maxLambda;  clear maxLambda Kappa;     
        % construct regularizer R(x)
        R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, KapPatch, mOmega, numBlock, CluInt);
            
end

fprintf('beta = %.1e, gamma = %g: \n\n', beta, gamma);

info = struct('intensity',I0,'ImgSiz',ImgSiz,'PatSiz',PatSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
    'nblock',nblock,'nIter',nIter,'CluInt',CluInt,'pixmax',pixmax,'transform',mOmega,...
    'xrla',[],'vIdx',[],'ClusterMap',[],'RMSE',[],'SSIM',[],'relE',[],'perc',[],'cost',[]);
%% Recon
SqrtPixNum = sqrt(sum(mask_roi(:)>0));
stop_diff_tol = 1e-3; % HU
iterate_fig = figure(55);
idx_old = ones([1,PatNum],'single');

for ii=1:nOuterIter
    figure(iterate_fig); drawnow;
    xold = xroi;
    AAA(1,ii) = norm(xroi(mask_roi) - xtrue(mask_roi)) / SqrtPixNum;
    fprintf('RMSE = %g\n', AAA(1,ii));
    info.RMSE = AAA(1,:);
    AAA(2,ii) = ssim(xroi, xtrue);
    fprintf('SSIM = %g\n',  AAA(2,ii));
    info.SSIM = AAA(2,:);
    
    fprintf('Iteration = %d:\n', ii);
    [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(sino, '2d'),  reshaper(wi, '2d'),...
        R, denom, D_R, 'pixmax', inf, 'chat', 0, 'alpha', 1.999, 'rho', [],'niter', nIter);

    
    [info.perc(:,ii),info.vIdx] = R.nextOuterIter();
    fprintf('perc = %g\n', info.perc(:,ii));

    idx_diff = idx_old - info.vIdx;
    fprintf('Idx Change Perc = %g\n', nnz(idx_diff)/PatNum);
    idx_old = info.vIdx;
    
%     info.cost(:,ii) = cost;
    info.xrla = ig.embed(xrla_msk); xroi = info.xrla(:,:,start_slice:end_slice);
    info.relE(:,ii) =  norm(xroi(mask_roi) - xold(mask_roi)) / SqrtPixNum;
    fprintf('relE = %g\n', info.relE(:,ii));
    if info.relE(:,ii) < stop_diff_tol
        break
    end
    figure(120), imshow(xroi(:,:,end/2), [800 1200]); drawnow;
    %      figure(120), im('mid3','notick',permute(info.xrla,[2 1 3]),[800,1200]); drawnow;
    
%     if (mod(ii,CluInt) ==1 )
%         info.ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numBlock);
%         figure(iterate_fig); drawnow;
%         central_map = info.ClusterMap(:,:,end/2);
%         central_xrla = info.xrla(:,:,end/2);
%         subplot(2,3,1),imshow((central_map == 1) .* central_xrla, [800,1200]);
%         subplot(2,3,2);imshow((central_map == 2) .* central_xrla, [800,1200]);
%         subplot(2,3,3);imshow((central_map == 3) .* central_xrla, [800,1200]);
%         subplot(2,3,4);imshow((central_map == 4) .* central_xrla, [800,1200]);
%         subplot(2,3,5);imshow((central_map == 5) .* central_xrla, [800,1200]);
%     end
end

%% Calculate clusterMap
% if(isCluMap == 1)
%     info.ClusterMap = ClusterMap(size(info.xrla), info.PatSiz, info.SldDist, info.vIdx, PatNum, numBlock);
%     %    C = CalClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numBlock);
%     figure(55), im('mid3',permute(info.ClusterMap,[2 1 3]), [1,numBlock]), cbar
%     %  figure(55), im('mid3',permute((info.ClusterMap == numBlock).*info.xrla,[2 1 3]), [800,1200]), cbar
% end
%%
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

% xrla = ig.embed(xrla_msk);
% figure name 'compare'
% imshow(cat(2, info.xrla(:,:,end/2), xrlalm(:,:,end/2)), [800 1200]);colorbar;

% im('mid3','notick', permute(info.xrla,[2 1 3]),[800,1200])

% save('info.mat', 'info')
% export_fig x.pdf -transparent

