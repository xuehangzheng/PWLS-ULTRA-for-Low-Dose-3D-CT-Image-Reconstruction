%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear; close all; 
addpath(genpath('../toolbox'));

%% setup target geometry and weight
down = 1; % downsample rate
cg = ct_geom('ge2', 'down', down);  

ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
ig.mask = ig.circ > 0; % can be omitted
A = Gcone(cg, ig, 'type', 'sf2', 'nthread', jf('ncore')*2-1);
% if neccessary, one could modify maxNumComThreads/jf('ncore') to make full
% use of threads of your machine to accelerate the computation of 
% forward and back projections. 

%% load external parameter
I0 = 1e4; % photon intensity

dir = ['../data/3Dxcat/tmp/' num2str(I0)];

fprintf('Loading sinogram, weighting, kappa...\n');
load([dir '/sino_cone.mat']);
load([dir '/wi.mat']);
load([dir '/kappa.mat']);
load([dir '/denom.mat']);

fprintf('Loading xfdk...\n'); 
load([dir '/xfdk.mat']);

%% setup edge-preserving regularizer
% set up ROI
% roi = ig.mask; start_slice = 17; end_slice = 80;
% roi(:,:,1:start_slice-1) = 0; roi(:,:,end_slice+1:end) = 0; 
% roi = roi(ig.mask);
load('xtrue_crop17-80.mat'); % ground truth in the ROI

nIter = 50;
nblock = 24; 
l2b = 14.5;

Ab = Gblock(A, nblock); clear A

delta = 1e1;  % 10 HU
pot_arg = {'lange3', delta};   % potential function

b1 = 1/ig.dx^2; b2 = 1/(ig.dx^2+ig.dy^2);
b3 = 1/ig.dz^2; b4 = 1/(ig.dx^2+ig.dz^2);
b5 = 1/(ig.dx^2+ig.dy^2+ig.dz^2);
beta = 2^l2b*[b1 b1 b2 b2 b5 b4 b5 b4 b3 b4 b5 b4 b5];
R = Reg1(sqrt(kappa), 'type_penal','mex','offsets','3d:26','beta',beta,... 
        'pot_arg', pot_arg, 'distance_power', 0,'nthread', jf('ncore')*2-1, 'mask',ig.mask);
        % sqrt(kappa) -- achieve uniform noise
        % kappa -- achieve uniform resolution

% check fwhm 
% fprintf('calculating fwhm...\n');
%  [~,~,fwhm,~,~] = qpwls_psf(Ab, R, 1, ig.mask, Gdiag(wi), 'fwhmtype', 'profile'); 

%% Recon

fprintf('iteration begins...\n'); 
[xrlalm_msk, info] = pwls_ep_os_rlalm(xfdk(ig.mask), Ab, reshaper(sino, '2d'), R, ...
             'wi',reshaper(wi, '2d'), 'niter', nIter, 'denom',denom,...
             'chat', 0, 'xtrue', xtrue, 'mask', ig.mask, 'isave', 'last');

SSIM = info.SSIM;  
RMSE = info.RMSE;
figure name 'RMSE'
plot(RMSE,'-*')
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
legend('Axial-PWLS-EP')

xrlalm = ig.embed(xrlalm_msk);
% figure name 'xrlalm';
% imshow(cat(2, xrlalm(:,:,end/2), xfdk(:,:,end/2)), [800 1200]); colorbar;

figure;im('mid3','notick', permute(xrlalm,[2 1 3]),[800,1200]),cbar
