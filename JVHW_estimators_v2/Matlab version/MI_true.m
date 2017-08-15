function MI = MI_true(pxy)
%MI_true  computes mutual information I(X;Y) given the joint distribution
% 
% Input:
% ----- pxy: Sx-by-Sy matrix capturing the joint probability masses of the
%            the bivariate RVs (X,Y), where Sx and Sy are the support size 
%            of X and Y, respectively. The (i,j)-th entry of pxy denotes
%            the joint probability Pr(X = i,Y = j).
%
% Output:
% ----- MI: the mutual information I(X;Y), which is a scaler. 
   
	% Error-check of the input distribution 
    if any(imag(pxy(:))) || any(isinf(pxy(:))) || any(isnan(pxy(:))) || any(pxy(:) < 0) || any(pxy(:) > 1)
        error('The probability elements must be real numbers between 0 and 1.');
    elseif abs(sum(pxy(:)) - 1) > sqrt(eps)
        error('Sum of the probability elements must equal 1.');
    end

    % Calculate marginals of X and Y
    px = sum(pxy,2);   
    py = sum(pxy,1);   

    % I(X;Y) = H(X) + H(Y) - H(X,Y)
    MI = entropy_true(px) + entropy_true(py) - entropy_true(pxy(:));
end