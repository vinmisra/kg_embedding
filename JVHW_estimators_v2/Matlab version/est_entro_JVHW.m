function est = est_entro_JVHW(samp)
%est_entro_JVHW  Proposed JVHW estimate of Shannon entropy (in bits) of the
%                input sample
%
% This function returns a scalar JVHW estimate of the entropy of samp when 
% samp is a vector, or returns a row vector consisting of the JVHW estimate
% of each column of samp when samp is a matrix.
%
% Input:
% ----- samp: a vector or matrix which can only contain integers. The input
%             data type can be any interger classes such as uint32/uint64,
%             or floating-point such as single/double. 
% Output:
% ----- est: the entropy (in bits) of the input vector or that of each 
%            column of the input matrix. The output data type is double.     

    if ~isequal(samp, fix(samp))
        error('Input sample must only contain integers!');
    end

    if isrow(samp)
        samp = samp.';
    end
    [n, wid] = size(samp);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    order = 4 + ceil(1.2*log(n));
    V = [0.3303  -0.3295  0.4679];

    persistent poly_coeff_r;
    if isempty(poly_coeff_r)
        load poly_coeff_r.mat poly_coeff_r;
    end
    coeff = poly_coeff_r{order};  % coefficients in increasing powers of the best polynomial approximation 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % A fast algorithm to compute the fingerprint (histogram of histogram) along each column of samp
    f = find([diff(sort(samp)); ones(1,wid,class(samp))]); % f: linear index of the last occurrence in each set of repeated values along the columns of samp
    f = accumarray({[f(1);diff(f)],ceil(f/n)},1);   % f: fingerprint
    
    prob = (1:size(f,1))/n;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    f1nonzero = find(f(1,:) > 0);
    lenf1nonzero = length(f1nonzero);
    c_1 = zeros(1, wid);
    if n > 15 && lenf1nonzero >0
        c_1(f1nonzero) = V * [ log(n) * ones(1,lenf1nonzero); log(f(1,f1nonzero)); ones(1,lenf1nonzero)];
        c_1 = max(c_1, 1/(1.9*log(n))); % make sure threshold is higher than 1/n
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    prob_mat = entro_mat(prob, n, coeff, c_1);
    est = sum(f.*prob_mat, 1)/log(2);
end 


function output = entro_mat(x, n, g_coeff, c_1)
    K = length(g_coeff) - 1;   % K: the order of best polynomial approximation, g_coeff = {g0, g1, g2, ..., g_K}
    thres = 4*c_1*log(n)/n;
    output = zeros(length(x), length(c_1));
    [thres, x] = meshgrid(thres,x);   
    region_large = x>thres;
    region_nonlarge = ~region_large;
    region_mid = x>thres/2 & region_nonlarge;
    output(region_large) = -x(region_large) .* log(x(region_large)) + 1/(2*n);   
    if nnz(region_nonlarge)
        x1(:,1) = x(region_nonlarge);  
        thres1(:,1) = thres(region_nonlarge); % Ensure x1 and thres1 are column vectors
        q = 0:K-1;
        output(region_nonlarge) = cumprod([thres1, bsxfun(@minus, n*x1, q)./bsxfun(@times, thres1, n-q)],2)*g_coeff.'- x1.*log(thres1);
    end
    ratio = 2*x(region_mid)./thres(region_mid) - 1;
    output(region_mid) = ratio.*(-x(region_mid) .* log(x(region_mid)) + 1/(2*n)) + (1-ratio).*output(region_mid); 
    output = max(output,0);
end


