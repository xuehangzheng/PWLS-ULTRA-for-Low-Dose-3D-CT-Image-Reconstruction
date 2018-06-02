%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all; 
addpath(genpath('../toolbox'));

%% setup target geometry and weight
down = 1; % downsample rate
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);

ig = image_geom('nx', 420, 'dx', 500/512, 'down', down);
ig.mask = ig.circ > 0;
A = Gtomo2_dscmex(sg, ig);
%% load external parameter
I0 = 1e4; % photon intensity
dir = ['../data/2Dxcat/tmp/' num2str(I0)];

fprintf('Loading sinogram, weight, kappa, fbp...\n');
load([dir '/sino_fan.mat']);
load([dir '/wi.mat']);
load([dir '/kappa.mat']);
load([dir '/denom.mat']);

load([dir '/xfbp.mat']);
% figure name 'xfbp'
% imshow(xfbp, [800 1200]);

%% setup edge-preserving regularizer
nIter = 50;
nblock = 24; 
l2b = 16;

delta = 1e1; % 10 HU
pot_arg = {'lange3', delta};  % potential function
R = Reg1(sqrt(kappa), 'beta', 2^l2b, 'pot_arg', pot_arg, 'nthread', jf('ncore'));

% fprintf('calculating fwhm...\n');
% [~,~,fwhm,~,~] = qpwls_psf(A, R, 1, ig.mask, Gdiag(wi), 'fwhmtype', 'profile');

%% Recon 
load('slice420.mat');

fprintf('iteration begins...\n'); 
[xrlalm_msk , info] = pwls_ep_os_rlalm_2d(xfbp(ig.mask), A, sino, R, 'wi', wi, ...
            'pixmax', inf, 'isave', 'last',  'niter', nIter, 'nblock', nblock, ...
            'chat', 0, 'denom',denom, 'xtrue', xtrue, 'mask', ig.mask);
        
AAA(1,:) = info.RMSE;
AAA(2,:) = info.SSIM;  
figure name 'RMSE'
plot(info.RMSE,'-*')
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
legend('PWLS-EP')


xrlalm = ig.embed(xrlalm_msk);
figure name 'xrlalm'
imshow(cat(2, xrlalm(:,:,end), xfbp), [800 1200]);colorbar;

% save('xrlalm.mat','xrlalm')
% save('AAA.mat', 'AAA')
% export_fig x.pdf -transparent