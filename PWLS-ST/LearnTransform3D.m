%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('../toolbox'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('../data/3Dxcat/phantom154.mat');
% load('../data/3Dxcat/phantom_crop154.mat');
% could also use air-cropped phantom to save some computation


PatSiz = [8 8 8];       % patch size
SldDist = 2 * [1 1 1];  % sliding distance

lambada0  = 31;         % set it large enough to guarantee well-conditioned
iter = 1000;            % iteration
gamma = 110;
nz = 54;                % number of training slices

for ii=1:nz % I used #1-96 as testing data
  image(:,:,ii) = downsample2(phantom(:,:,ii+100), 2);
end
clear phantom;

fprintf('extracting patches...\n')
patch = im2colstep(single(image), PatSiz, SldDist); clear image;
patch = double(patch);
fprintf('Length of training set: %d\n', size(patch, 2));


D1 = dctmtx(PatSiz(1));
D2 = dctmtx(PatSiz(2));
D3 = dctmtx(PatSiz(3));
mOmega = kron(kron(D1, D2), D3);  % 3D DCT Initialization. Be careful of the order of D1 D2 D3!
clear D1 D2 D3

lambada = lambada0 * norm(patch,'fro')^2;
[U,S,V] = svd((patch*patch') + lambada*eye(prod(PatSiz)));
LL2 = (inv(U*(S^(1/2))*V')); clear U S V

perc = zeros(iter,1,'single'); % sparsity (percentage)
for j=1:iter
  fprintf('iteration = %d:\n', j);
  % sparse coding
  sparseCode = mOmega * patch;
  % hard-thresholding
  sparseCode = sparseCode.*(abs(sparseCode) > gamma);
  perc(j) = nnz(sparseCode)/ numel(sparseCode) * 100;
  fprintf('sparsity  = %g\n', perc(j));
  % transform update
  [Q1,Si,R] = svd(LL2*patch*sparseCode');
  sig = diag(Si);
  gamm = (1/2)*(sig + (sqrt((sig.^2) + 2*lambada)));
  B = R*(diag(gamm))*Q1';
  mOmega = B*(LL2);
end

%%%%%%%%%%%%%%%%%%%%%% check convergency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(); plot(perc);
xlabel('Number of Iteration','fontsize',18)
ylabel('Sparity ( % )','fontsize',18)

condTransform = cond(mOmega); % check condition number

%%%%%%%%%%%%%%%%%%%%% visualize learned transforms %%%%%%%%%%%%%%%%%%%%%%%
transform = mOmega(1:256,1:64);

ka = 16-1;
% normalization
for i=1:size(transform,1)
    row = transform(i,:);
    row = row - min(row);
    if(max(row)>0)
      row = row / (max(row));
    end
    transform(i,:) = row;
end
   
kb = ka;  counter = 1; 
jy = 1; % control the line interval between different patches

Ta = zeros((8+jy)*ka+8,(8+jy)*kb+8); % use "ones" for different looking
for i=1:8+jy:ka*(8+jy)+1
    for j=1:8+jy:(kb*(8+jy))+1
       Ta(i:i+7,j:j+7) = reshape(transform(counter,:),8,8);
       counter=counter+1;
    end
end
figure;imagesc(Ta);colormap('Gray');axis off;axis image; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
