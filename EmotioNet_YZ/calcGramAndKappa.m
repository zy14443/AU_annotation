% Samuel Rivera
% Jan 30, 2011
%
% notes: calc gram matrix and kappa  vector ( kernel inner product between
% test sample and all training samles in gram matrix, 
%
% X in (p x N)
% x in (p x Ntest)
% kernelType:  1 for rbf
%              2 for softmax
%


function [K kappa ] = calcGramAndKappa( X, x, kernelType, params )


% rbf kernel if 1, or empty
if kernelType == 1 || isempty( kernelType) 
    
    sigma = params.sigma;
    
    % calculate the pairwise distance matrix
    nAll = size( [X x],2);
    A = [X x]'*[X x];
    dA = diag(A);
    DD = repmat(dA,1,nAll) + repmat(dA',nAll,1) - 2*A;
    tempG = exp( DD./(-1*2*sigma.^2));

    n = size( X,2);

    K = tempG( 1:n, 1:n);  % N x N
    kappa = tempG( 1:n, n+1:end);  % n x Ntest

    
elseif kernelType == 2  % softmax kernel 
    
    % original rbf kernel
    [K kappa] = calcGramAndKappa( X, x, 1, params );
    
    % normalize
    kappa = kappa./repmat(sum(K,2), [ 1, size(kappa,2)] ); 
    K = K./repmat( sum(K,1), [ size(K,1),1] );
    
    
else
    error( 'SR: undefined kernelType' );
    
end
    
    