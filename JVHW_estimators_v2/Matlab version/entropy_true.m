function H = entropy_true(p)
%entropy_true  computes Shannon entropy H(p) (in bits) of the input discrete
%              distribution.
%
% This function returns a scalr entropy when the input distribution p is a 
% vector, or returns a row vector containing the entropy of each column of 
% the input probability matrix p. 

    % Error-check of the input distribution
    if any(imag(p(:))) || any(isinf(p(:))) || any(isnan(p(:))) || any(p(:) < 0) || any(p(:) > 1)
        error('The probability elements must be real numbers between 0 and 1.');
    elseif any(abs(sum(p) - 1) > sqrt(eps))
        error('Sum of the probability elements must equal 1.');
    end

    H = sum(-p.*log2(p),'omitnan');  % Use 'omitnan' option to skip zero probability, if any, in p 
end