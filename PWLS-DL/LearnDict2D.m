clear; close all; 
addpath(genpath('../data/2Dxcat'));
addpath(genpath('../toolbox'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('../data/3Dxcat/phantom154.mat');

  PatSiz = 8 * [1 1];  % patch size
 SldDist = 1 * [1 1];  % sliding distance


patch = [];
for ii=[10 30 40 50 80]  % training slices
%    for ii = 48          % testing slice
   image = downsample2(phantom(:, :, ii), 2); 
   patch_tmp = im2colstep(single(image), PatSiz, SldDist);   
   patch = [patch patch_tmp];
end
clear phantom image patch_tmp
fprintf('Length of training set: %d\n', size(patch, 2));
 
%% 2D DCT Initialization.  %%

D = kron(dctmtx(8),dctmtx(8));% DCT Initialization

K = 64 * 4;  % 4-fold overcomplete
Pn = ceil(sqrt(K));
bb = 8;
DCT = zeros(bb,Pn);
for k = 0:1:Pn-1
    V = cos([0:1:bb-1]'*k*pi/Pn);
    if k > 0 
        V = V - mean(V); 
    end
    DCT(:,k+1) = V / norm(V);
end
D = kron(DCT, DCT);

%% run k-svd training %%
% min  |Gamma|_0   s.t.  |X_i - D*Gamma_i|_2 <= EPSILON, |Gamma_i|_0 <= T

params.data = double(patch);  clear patch
% params.Tdata = 8;  % sparsity-based ksvd
params.Edata = 1e-1; % error-based ksvd
params.codemode = 'error';
params.maxatoms = 20;
params.initdict = D;
% params.dictsize = prod(PatSiz) * 4;
params.iternum = 1000;
params.memusage = 'high';
% params.checkdict = 'off';
[mOmega,Gamma,spar_err,sparsity] = ksvd(params);
% r - number of replaced atoms
% t - target function value (and its value on the test data if provided)
info.mOmega = mOmega;
info.spar_err = spar_err;
info.sparsity = sparsity;
%% show results %%

figure;title('K-SVD error convergence');
plot(sparsity); xlabel('Iteration'); ylabel('sparsity');

dictimg = showdict(mOmega, PatSiz ,round(sqrt(K)),round(sqrt(K)),'lines','highcontrast');
figure; imshow(imresize(dictimg,2,'nearest'));
