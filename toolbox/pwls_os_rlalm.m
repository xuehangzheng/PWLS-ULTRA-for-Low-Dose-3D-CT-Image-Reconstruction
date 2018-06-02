 function [x, cost] = pwls_os_rlalm(x, Ab, yi, wi, R, denom, D_R, varargin)
%function [xs, info] = ir_pwls_os_rlalm(x, Ab, yi, R, [options])
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|	x	  [np 1]		initial estimate
%|	Ab	[nd np]		Gblock object (needs abs(Ab) method)
%|			         	or sparse matrix (implies nsubset=1)
%|	yi	[nb na]		measurements (noisy sinogram data)
%|	wi	[nb na]		weighting sinogram (default: [] for uniform)
%|	R	penalty		object (see Reg1.m), can be []
%|	denom	[np 1]		precomputed denominator
%|  D_R [np 1]  diagonal majorizing matrix of the Hessian of R

%| option
%|	niter			# of iterations (default: 1)
%|	pixmax	[1] or [2]	max pixel value, or [min max] (default [0 inf])
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
%|	x	[np niter]	iterates
%|	cost	[niter 1]	cost values
%|
%| 2015-11-30, Hung Nien, based on ir_pwls_os_lalm
%| 2015-11-30, tweaks by Jeff Fessler
%| 2016, modified by Xuehang Zheng

if nargin == 1 && streq(x, 'test'), ir_pwls_os_rlalm_test, return, end
if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
% arg.isave = [];
arg.userfun = @userfun_default;
arg.userarg = {};
arg.pixmax = inf;
arg.chat = false;
% arg.aai = [];
arg.rho = []; % default: decreasing rho
arg.alpha = 1.999;
arg.relax0 = 1;
% arg.denom = [];
arg.scale_nblock = true; % traditional scaling
arg.update_even_if_denom_0 = true;
arg = vararg_pair(arg, varargin);

% arg.isave = iter_saver(arg.isave, arg.niter);

Ab = block_op(Ab, 'ensure'); % make it a block object (if not already)
nblock = block_op(Ab, 'n');
starts = subset_start(nblock);

cpu etic

if isempty(wi)
	wi = ones(size(yi), class(yi));
end

% if isempty(arg.aai)
% 	aai = reshape(sum(abs(Ab)'), size(yi)); % a_i = sum_j |a_ij|
% end

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


if ~arg.update_even_if_denom_0
	% todo: this may not work for LALM because "denom" appears in numerator!
	denom(denom == 0) = inf; % trick: prevents pixels where denom=0 being updated
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
% np = length(x);
% xs = zeros(np, length(arg.isave), 'single');
% if any(arg.isave == 0)
% 	xs(:, arg.isave == 0) = single(x);
% end

%info = zeros(niter,?); % do not initialize since size may change

% initialization
iblock = nblock;
ia = iblock:nblock:na;
li = Ab{iblock} * x;
li = reshape(li, nb, length(ia));
resid = wi(:,ia) .* (li - yi(:,ia));
zeta = nblock * Ab{iblock}' * resid(:);

g = rho(1) * zeta;
h = denom .* x - zeta;
  

% iterate
for iter = 1:arg.niter
	ticker(mfilename, iter, arg.niter)

	relax = relax0 / (1 + relax_rate * (iter-1));
  
	% loop over subsets
	for iset = 1:nblock
		k = nblock*(iter-1)+iset;

		num = rho(k) * (denom .* x - h) + (1-rho(k)) * g;
		den = rho(k) * denom;

	    num = num + R.cgrad(x);
        den = den + D_R;

		x = x - relax * num ./ den;
		x = max(x, pixmin);
		x = min(x, pixmax);

		iblock = starts(iset);
		ia = iblock:nblock:na;

		li = Ab{iblock} * x;
		li = reshape(li, nb, length(ia));
		resid = wi(:,ia) .* (li - yi(:,ia));

		zeta = nblock * Ab{iblock}' * resid(:); % A' * W * (y - A*x)
		g = (rho(k) * (alpha * zeta + (1-alpha) * g) + g) / (rho(k)+1);
		h = alpha * (denom .* x - zeta) + (1-alpha) * h; 
    
    end
    
%     to decide how many inner iterations should be used 
%     cost = R.penal(Ab, x, wi, yi);
      
end

if arg.chat
  % calculate the cost value for each outer iteration
   cost = R.penal(Ab, x, wi, yi);
else
   cost = 0;
end

