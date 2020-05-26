%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('~/Desktop/data/2Dxcat'));
addpath(genpath('~/Desktop/toolbox'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('../data/3Dxcat/phantom154.mat');

l = 64;          % patch size
stride = 1;      % overlapping stride
lambada0 = 31;   % set it large enough to guarantee well-conditioned
iter = 1000;     % iteration
gamma = 110;

patch=[];
for ii=[10 30 40 50 80]  % training data
    %  for ii = 48     % testing data
    image = downsample2(phantom(:, :, ii), 2);
    % The Mathworks 'im2col' is quicker but only for stride 1.
    %    patch_tmp = im2col(image, sqrt(l) * [1 1], 'sliding');
    %     [patch_tmp, ~] = image2patch(image,sqrt(l) * [1 1], stride);
    patch_tmp = im2colstep(single(image), sqrt(l)*[1 1], stride*[1 1]);
    patch = [patch patch_tmp];
end
fprintf('Length of training set: %d\n', size(patch, 2));
patch = double(patch);

mOmega = kron(dctmtx(sqrt(l)),dctmtx(sqrt(l)));% DCT Initialization
lambada = lambada0 * norm(patch,'fro')^2;

%[U,S,V] = svd((patch*patch') + (0.5*lambada*eye(l)));
[U,S,V] = svd((patch*patch') + lambada*eye(l));
LL2 = (inv(U*(S^(1/2))*V'));

perc = zeros(iter,1,'single'); % sparsity (percentage)
for j=1:iter
    fprintf('iteration = %d:\n', j);
    % sparse coding
    sparseCode = mOmega * patch;
    sparseCode = sparseCode.*(abs(sparseCode) > gamma);
    perc(j) = nnz(sparseCode)/ numel(sparseCode) * 100;
    fprintf('sparsity  = %g:\n', perc(j));
    % transform update
    [Q1,Si,R] = svd(LL2*patch*sparseCode');
    sig = diag(Si);
    gamm = (1/2)*(sig + (sqrt((sig.^2) + 2*lambada)));
    B = R*(diag(gamm))*Q1';
    mOmega = B*(LL2);
end

% save('mOmega.mat','mOmega')
% save('perc.mat','perc')

% check convergency
figure(); plot(perc);
xlabel('Number of Iteration','fontsize',18)
ylabel('Sparity ( % )','fontsize',18)
% check condition number
% condTrans = cond(transform);

% show learnt transform
for i=1:size(mOmega,1)
    mOmega(i,:)=mOmega(i,:)-min(mOmega(i,:));
    if(max(mOmega(i,:))>0)
        mOmega(i,:)=mOmega(i,:)/(max(mOmega(i,:)));
    end
end

jy=1;cc=1;
Ta=(max(max(mOmega)))*ones((8+jy)*7 + 8,(8+jy)*7 + 8);
for i=1:8+jy:(7*(8+jy))+1
    for j=1:8+jy:(7*(8+jy))+1
        Ta(i:i+7,j:j+7)=reshape((mOmega(cc,:))',8,8);
        cc=cc+1;
    end
end
figure();imagesc(Ta);colormap('Gray');axis off;axis image;

