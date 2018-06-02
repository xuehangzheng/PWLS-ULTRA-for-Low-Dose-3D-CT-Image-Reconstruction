%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear; close all;
addpath(genpath('../data/3Dxcat'));
addpath(genpath('../toolbox'));

%% setup target geometry and weight
down = 1; % downsample rate
cg = ct_geom('ge2', 'down', down);
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
ig.mask = ig.circ > 0;
A = Gcone(cg, ig, 'type', 'sf2','nthread', jf('ncore')*2-1);
% if neccessary, one could modify maxNumComThreads/jf('ncore') to make full
% use of threads of your machine to accelerate the computation of 
% forward and back projections. 

%% load external data
I0 = 1e4; % photon intensity

% load PWLS-EP Recon as initialization: xrlalm
 load('1e4_l2b14dot5_os24_iter50.mat'); % change intial EP image when I0 is changed!
% load('5e3_l2b14dot5_os24_iter50.mat');

% load transform: mOmega
 load('slice101_154_block1_SldDist2_iter1000_gamma50.mat');
mOmega = info.mOmega; clear info

% load ground truth image: xtrue
load('xtrue_crop17-80.mat');

% load measurements and initial data
dir = ['../data/3Dxcat/tmp/' num2str(I0)];
printm('Loading sinogram, weighting, xfdk...');
load([dir '/sino_cone.mat']);
load([dir '/wi.mat']);
% load([dir '/xfdk.mat']);
% figure; imshow(xfdk(:,:,end/2), [800 1200]);

%% setup edge-preserving regularizer
xrla_msk = xrlalm(ig.mask);  % initial EP image
%xrlalm = xfdk .* ig.mask;     % initial FDK image
%xrla_msk = xfdk(ig.mask);

% set up ROI
start_slice = 17; end_slice = 80;
xroi = xrlalm(:,:,start_slice:end_slice); clear xrlalm
mask_roi = ig.mask(:,:,start_slice:end_slice);
% roi = ig.mask; roi(:,:,1:start_slice-1) = 0; roi(:,:,end_slice+1:end) = 0;
% roi = roi(ig.mask);

ImgSiz = [ig.nx ig.ny ig.nz];  % image size
PatSiz = [8 8 8];          % patch size
SldDist = 2 * [1 1 1];         % sliding distance

nblock = 4;                    % Subset Number
nOuterIter= 10;             % T--Outer Iteration
nIter = 2;                     % I--Inner Iteration
pixmax = inf;                  % Set upper bond for pixel values

printm('Pre-calculating denominator: D_A...');
% denom = abs(A)' * col(reshape(sum(abs(A)'), size(wi)) .* wi);
% denom= abs(A)'*(wi(:).*(abs(A)*ones(size(xrla_msk,1),1)));
load([dir '/denom.mat']);

Ab = Gblock(A, nblock); clear A
% pre-compute D_R
maxLambda = max(eig(mOmega' * mOmega));

KapType = 0;

switch KapType
    
    case 0
        beta= 2e5;
        gamma = 18;
        
        PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
        KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);
        D_R = 2 * beta * KK(ig.mask) * maxLambda; clear PP KK
        % D_R = 2 * beta * prod(PatSiz)/ prod(SldDist) * maxLambda;
        R = Reg_ST(ig.mask, PatSiz, ImgSiz, SldDist, beta, gamma, mOmega);
        
    case 1
        beta = 1e4;
        gamma = 20;
        
        load([dir '/kappa.mat']);
        KapPatch = im2colstep(kappa, PatSiz, SldDist); clear kappa;
        KapPatch = mean(KapPatch,1);   % l1 norm
        % KapPatch = max(KapPatch,[],1); % inf norm
        % KapPatch = sqrt(sum(KapPatch.^2,1));   % l2 norm
        KapPatch = repmat(KapPatch, prod(PatSiz), 1);
        Kappa = col2imstep(single(KapPatch), ImgSiz, PatSiz, SldDist);
        D_R = 2 * beta * Kappa(ig.mask) * maxLambda;  clear maxLambda Kappa;        
        % construct regularizer R(x)
        R = Reg_ST_Kappa(ig.mask, PatSiz, ImgSiz, SldDist, beta, gamma, KapPatch, mOmega);
        clear KapPatch
        
end

info = struct('intensity',I0,'PatSiz',PatSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
    'nblock',nblock,'nIter',nIter,'pixmax',pixmax,'transform',mOmega,...
    'xrla',[],'RMSE',[],'SSIM',[],'relE',[],'perc',[],'cost',[]);

%% Recon
SqrtPixNum = sqrt(sum(mask_roi(:)>0));
stop_diff_tol = 1e-3; % HU

for ii=1:nOuterIter
    xold = xroi;
    AAA(1,ii) = norm(xroi(mask_roi) - xtrue(mask_roi)) / SqrtPixNum;
    fprintf('RMSE = %g\n', AAA(1,ii));
    info.RMSE = AAA(1,:);
    AAA(2,ii) = ssim(xroi, xtrue);
    fprintf('SSIM = %g\n',  AAA(2,ii));
    info.SSIM = AAA(2,:);
    
    fprintf('Iteration = %d:\n', ii);
    [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(sino,'2d'), reshaper(wi,'2d'),  ...
        R, denom, D_R, 'pixmax', pixmax, 'chat', 0, 'alpha', 1.999, 'rho', [], 'niter', nIter);
    
    info.perc(:,ii) = R.nextOuterIter();
    fprintf('perc = %g\n', info.perc(:,ii));
    
    %     info.cost(:,ii) = cost';
    info.xrla = ig.embed(xrla_msk); xroi = info.xrla(:,:,start_slice:end_slice);
    
    info.relE(:,ii) =  norm(xroi(mask_roi) - xold(mask_roi)) / SqrtPixNum;
    fprintf('relE = %g\n', info.relE(:,ii));
    if info.relE(:,ii) < stop_diff_tol
        break
    end
    
    figure(110), imshow(xroi(:,:,end/2), [800 1200]); drawnow;
    %      figure(110), im('mid3','notick',permute(info.xrla,[2 1 3]),[800,1200]); drawnow;
    
end

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
legend('PWLS-ST')


% xrla = ig.embed(xrla_msk);
% figure name 'compare'
% imshow(cat(2, info.xrla(:,:,end/2), xrlalm(:,:,end/2)), [800 1200]);colorbar;

% im('mid3','notick',permute(info.xrla,[2 1 3]),[800,1200])

% save('info.mat', 'info')

