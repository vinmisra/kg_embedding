function x = randsmpl(p, m, n)
%RANDSMPL  Generate random matrix with i.i.d. samples according to a given
%          discrete distribution
%
%   x = randsmpl(p, m, n) retruns an m-by-n random integer matrix whose
%   elements are independently drawn from the discrete distribution with
%   the probability mass function p. Suppose the sample space comprises K
%   distinct outcomes, then p must be a (row or column) vector containing K
%   probability values adding up to 1, each measuring the probability of a
%   particular outcome. For example, x(i,j) = k means that the k-th outcome
%   occurrs in the (i,j)-th trial.
%
%   Remarks:
%       - The basic idea is to divide the interval [0,1] into K disjoint
%         bins, each with the length proportional to the probability mass
%         in p. Then, we draw unfiorm random samples from the U(0,1)
%         distribution, and determine the indices of the bins to which the
%         random samples belong.
%       - Note that histcounts (introduced in R2014b) and histc find not
%         only the indices of the histogram bins, but also the number of
%         samples belonging to each bin (i.e., the histogram). In contrast,
%         discretize (introduced in R2015a) is optimized solely for finding
%         the indices of the bins, thus it's more efficient than histcounts
%         and histc. For better performance, we use discretize here.
%       - We also found that discretize is not only faster, but also more
%         efficient in terms of memory consuption than histc. discretizemex
%         is the binary mex file of discretize which is included in the
%         current folder.
%       - For those running older versions of MATLAB (R2014b or older), two 
%         alternatives are provided:
%         1) Use the provided mex file, discretizemex (included in the
%            current folder) instead of discretize;
%         2) Use interp1 instead: though slightly slower than discretize,
%            the performance penalty of interp1 is almost negligible.
%
%   See also RAND, RANDI, RANDN, RNG.
%
%   Peng Liu, Aug. 10, 2015

	if ~isvector(p)
        error('Input distribution p must be a vector.')
    end
    
    % Error-check of the input distribution 
    if any(imag(p)) || any(isinf(p)) || any(isnan(p)) || any(p < 0) || any(p > 1)
        error('The probability elements must be real numbers between 0 and 1.');
    elseif abs(sum(p) - 1) > sqrt(eps)
        error('Sum of the probability elements must equal 1.');
    elseif any(p < eps)
        p(p < eps) = [];  % Remove essentially zero probability
    end
    
    edges = [0; cumsum(p(:))];
    if abs(edges(end) - 1) > sqrt(eps)   % Deal with floating-point errors introduced by cumulative sum
        edges = edges/edges(end);
    end
    edges(end) = edges(end) + eps(1);    
    
%     x = int32(discretize(rand(m, n), edges));      % For R2015a or newer version
    x = int32(discretizemex(rand(m, n), edges));   % Alternative method if discretize dosen't work
%     x = int32(interp1(edges,1:length(edges),rand(m, n),'previous'));   % Alternative method if discretize/discretizemex don't work