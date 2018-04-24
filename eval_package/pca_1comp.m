function [Comp] = pca_1comp(X)
% Principal Componant Analysis (PCA).
%
% Input
%   X       -  sample matrix, n x dim
%
% Output
%   Comp    -  the dominant principal components, dim x 1
%
% History
%   create  -  Yan Zhang (yz-cnsdqz.github.io)
%   modified from  -  Feng Zhou (zhfe99@gmail.com), 11-01-2009

[dim, n] = size(X);

% subtract the mean
me = sum(X, 2) / n;
X = X - repmat(me, 1, n);

% spectral decomposition
Cmat = X*X';

[V,D] = eig(Cmat);
Comp = V(:,1);
% [U, S] = svd(X, 0);
% [Lamb, index] = sort(sum(S, 2), 'descend');
% Dire = U(:, index);
% 
% Comp = Dire' * X;
