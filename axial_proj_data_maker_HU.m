%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear; close all; 
%% generate noisy sinogram and statistical weighting
I0 = 1e4; % photon intensity

down = 1; % downsample rate
cg = ct_geom('ge2'); 

dir = ['./data/3Dxcat/tmp/' num2str(I0)];

load('./data/3Dxcat/phantom_crop154.mat');  % unit of the loaded phantom: HU
phantom = phantom(:,:,1:96); % extract 96 slices for testing

mm2HU = 1000 / 0.02;
fprintf('generating noiseless sino...\n');
ig_hi = image_geom('nx',840,'dx',500/1024,'nz',96,'dz',0.625,'down',down);
A_hi = Gcone(cg, ig_hi, 'type', 'sf2', 'nthread', jf('ncore')*2-1);  
sino_true = A_hi * phantom;  clear A_hi;

fprintf('adding noise...\n');
yi = poisson(I0 * exp(-sino_true ./ mm2HU), 0, 'factor', 0.4);
var = 5; 
ye = var.* randn(size(yi)); % Gaussian white noise ~ N(0,std^2)
k = 1;
zi = k * yi + ye;
error = 1/1e1;
zi = max(zi, error);   
sino = -log(zi ./(k*I0)) * mm2HU; 

wi = (zi.^2)./(k*zi + var^2);  
save([dir '/wi.mat'], 'wi');    
save([dir '/sino_cone.mat'], 'sino'); 
% figure name 'Noisy sinogram'
% imshow(sino, [2 40000]);

%% setup target geometry and fbp
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
ig.mask = ig.circ > 0;
A = Gcone(cg, ig, 'type', 'sf2','nthread', jf('ncore')*2-1);
fprintf('fdk...\n');
xfdk = feldkamp(cg,ig,sino,'window','hanning,0.5','w1cyl',1,'extrapolate_t',round(1.3*cg.nt/2));
xfdk = max(xfdk , 0);
save([dir '/xfdk.mat'], 'xfdk');
figure;im('mid3',permute(xfdk,[2 1 3]),[800,1200])

%% setup kappa
fprintf('calculating kappa...\n');
kappa = sqrt( div0(A' * wi, A' * ones(size(wi))) );
kappa = max(kappa, 0.01*max(col(kappa)));
save([dir '/kappa.mat'], 'kappa');

%% setup diag{A'WA1}
printm('Pre-calculating denominator D_A...');
denom = A' * col(reshape(sum(A'), size(wi)) .* wi); 
save([dir '/denom.mat'], 'denom');
