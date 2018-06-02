 function [xs, info] = pwls_ep_os_rlalm_2d(x, A, yi, R, varargin)
%function [xs, info] = ir_pwls_os_rlalm(x, Ab, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| using relaxed linearized augmented lagrangian method with
%| (optionally relaxed) ordered subsets. (relaxed OS-LALM)
%|
%| See ?. 201? IEEE T-MI by Hung Nien & J A Fessler
%| "Relaxed linearized algorithms for faster X-ray CT image reconstruction"
%|	http://dx.doi.org/10.1109/TMI.201?
%|
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|	x	[np 1]		initial estimate
%|	Ab	[nd np]		Gblock object (needs abs(Ab) method)
%|				or sparse matrix (implies nsubset=1)
%|	yi	[nb na]		measurements (noisy sinogram data)
%|	R	penalty		object (see Reg1.m), can be []
%|
%| option
%|	niter			# of iterations (default: 1)
%|	wi	[nb na]		weighting sinogram (default: [] for uniform)
%|	pixmax	[1] or [2]	max pixel value, or [min max] (default [0 inf])
%|	denom	[np 1]		precomputed denominator
%|	aai	[nb na]		precomputed row sums of |Ab|
%|	relax0	[1] or [2]	relax0 or (relax0, relax_rate) (default 1)
%|	rho			AL penalty parameter (default: [] for decreasing rho)
%|	alpha			over-relaxation parameter (default: 1.999)
%|	userfun	@		user-defined function handle (see default below)
%|					taking arguments (x, userarg{:})
%|	userarg	{}		user arguments to userfun (default {})
%|	chat
%|
%| out
%|	xs	[np niter]	iterates
%|	info	[niter 1]	time
%|
%| 2015-11-30, Hung Nien, based on ir_pwls_os_lalm
%| 2015-11-30, tweaks by Jeff Fessler
%| 2016-11-30, updated by Xuehang Zheng

if nargin == 1 && streq(x, 'test'), ir_pwls_os_rlalm_test, return, end
if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.nblock = 1;
arg.niter = 1;
arg.isave = [];
arg.userfun = @userfun_default;
arg.userarg = {};
arg.pixmax = inf;
arg.chat = false;
arg.wi = [];
arg.aai = [];
arg.rho = []; % default: decreasing rho
arg.alpha = 1.999;
arg.relax0 = 1;
arg.denom = [];
arg.scale_nblock = true; % traditional scaling
arg.update_even_if_denom_0 = true;
arg.xtrue = zeros(size(x));
arg.mask = ones(size(arg.xtrue));

arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);


Ab = Gblock(A, arg.nblock);
% Ab = block_op(Ab, 'ensure'); % make it a block object (if not already)
nblock = block_op(Ab, 'n');
starts = subset_start(nblock);

cpu etic

wi = arg.wi;
if isempty(wi)
	wi = ones(size(yi));
end
if isempty(arg.aai)
	arg.aai = reshape(sum(abs(Ab)'), size(yi)); % a_i = sum_j |a_ij|
end

% check input sinogram sizes for OS
if (ndims(yi) ~= 2) || (size(yi,2) == 1 && nblock > 1)
	fail 'bad yi size'
end
if (ndims(wi) ~= 2) || (size(wi,2) == 1 && nblock > 1)
	fail 'bad wi size'
end

relax0 = arg.relax0(1);
if length(arg.relax0) == 1
	relax_rate = 0;
elseif length(arg.relax0) == 2
	relax_rate = arg.relax0(2);
else
	error relax
end

if length(arg.pixmax) == 2
	pixmin = arg.pixmax(1);
	pixmax = arg.pixmax(2);
elseif length(arg.pixmax) == 1
	pixmin = 0;
	pixmax = arg.pixmax;
else
	error pixmax
end

% likelihood denom, if not provided
denom = arg.denom;
if isempty(denom)
	denom = abs(Ab)' * col(arg.aai .* wi); % needs abs(Ab)
end
if ~arg.update_even_if_denom_0
	% todo: this may not work for LALM because "denom" appears in numerator!
	denom(denom == 0) = inf; % trick: prevents pixels where denom=0 being updated
end

if isempty(R)
	pgrad = 0; % unregularized default
	Rdenom = 0;
end

alpha = arg.alpha;
if alpha<1 || alpha>2
	fail 'alpha should be between 1 and 2'
end

rho = arg.rho;
if isempty(rho)
	rho = @(k) pi/(alpha*k) * sqrt(1 - (pi/(2*(alpha*k)))^2) * (k>1) + (k==1);
else
	rho = @(k) rho; % constant user-specified value
end

[nb na] = size(yi);

x = x(:);
np = length(x);
xs = zeros(np, length(arg.isave), 'single');
if any(arg.isave == 0)
	xs(:, arg.isave == 0) = single(x);
end

%info = zeros(niter,?); % do not initialize since size may change

% initialization
iblock = nblock;
ia = iblock:nblock:na;
li = Ab{iblock} * x;
li = reshape(li, nb, length(ia));
resid = wi(:,ia) .* (li - yi(:,ia));
if arg.scale_nblock
	scale = nblock; % traditional way
else
	scale = na / numel(ia); % alternative - untested
end
zeta = scale * Ab{iblock}' * resid(:);

g = rho(1) * zeta;
h = denom .* x - zeta;

info = struct('cost',[], 'RMSE',[], 'relE',[], 'SSIM', []);
xtrue_msk = arg.xtrue(arg.mask);
SqrtPixNum = sqrt(sum(arg.mask(:)>0));
% iterate
for iter = 1:arg.niter
	ticker(mfilename, iter, arg.niter)
  xold = x;
	relax = relax0 / (1 + relax_rate * (iter-1));

  if isempty(arg.xtrue) ~= true
     info.RMSE(:,iter) = norm(x - xtrue_msk) / SqrtPixNum;
     disp(['RMSE = ', num2str(info.RMSE(:,iter))]);  
     info.SSIM(:,iter)= ssim(embed(x,arg.mask), arg.xtrue);
     disp(['SSIM = ', num2str(info.SSIM(:,iter))]);  
  end  
  
	% loop over subsets
	for iset = 1:nblock
		k = nblock*(iter-1)+iset;

		num = rho(k) * (denom .* x - h) + (1-rho(k)) * g;
		den = rho(k) * denom;
		if ~isempty(R)
			num = num + R.cgrad(R, x);
			den = den + R.denom(R, x);
		end
		x = x - relax * num ./ den;
		x = max(x, pixmin);
		x = min(x, pixmax);

		iblock = starts(iset);
		ia = iblock:nblock:na;

		li = Ab{iblock} * x;
		li = reshape(li, nb, length(ia));
		resid = wi(:,ia) .* (li - yi(:,ia));

		if arg.scale_nblock
			scale = nblock; % traditional way
		else
			scale = na / numel(ia); % alternative - untested
		end

		zeta = scale * Ab{iblock}' * resid(:); % A' * W * (y - A*x)
		g = (rho(k) * (alpha * zeta + (1-alpha) * g) + g) / (rho(k)+1);
		h = alpha * (denom .* x - zeta) + (1-alpha) * h;

  end
  

	if any(arg.isave == iter)
		xs(:, arg.isave == iter) = single(x);
	end

  if arg.chat   
     df = .5 * sum(col(wi) .* (Ab * x - col(yi)).^2, 'double');
     fprintf('df = %g\n', df); 
     rp = R.penal(R, x);
     fprintf('rp = %g\n', rp); 
	   info.cost(:,iter) = df + rp;
  end


  info.relE(:,iter) = norm(xold - x) / norm(x);   
  disp(['RelE = ', num2str( info.relE(:,iter))]);


  figure(20), imshow( embed(x, arg.mask), [800 1200]); 
end



% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%	x = evalin('caller', 'x');
	printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');


