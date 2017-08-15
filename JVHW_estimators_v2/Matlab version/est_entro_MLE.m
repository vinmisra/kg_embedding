function est = est_entro_MLE(samp)
%est_entro_MLE  Maximum likelihood estimate of Shannon entropy (in bits) of 
%               the input sample
%
% This function returns a scalar MLE of the entropy of samp when samp is a 
% vector, or returns a (row-) vector consisting of the MLE of the entropy 
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

    % A fast algorithm to compute the fingerprint (histogram of histogram) along each column of samp
    f = find([diff(sort(samp)); ones(1,wid,class(samp))]);
    f = accumarray({[f(1);diff(f)],ceil(f/n)},1);    % f: fingerprint

    prob = (1:size(f,1))/n;
    prob_mat = -prob.*log2(prob);
    est = prob_mat * f;
end