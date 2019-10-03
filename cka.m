function lCKA = cka(X,Y)
% function lCKA = cka(X,Y)
% Calculates the linear centered Kernel alignment across regions (or datasets)
% Based on Kornblith et al. (2019) bioRxiv paper (https://arxiv.org/abs/1905.00414)
%
% INPUT:
%           - X: nCond x nVox activation
%           - Y: nCond x nVox activation
% OUTPUT:
%           - lCKA: linear CKA calculated for region pair
%
% EBerlot, July 2019
% -------------------------------------------------------------------------
if size(X,1)~=size(Y,1)
    error('X and Y must have the same number of conditions (dimension 1)');
else
    lCKA = norm(X'*Y,'fro')^2 / (norm(X'*X,'fro')*norm(Y'*Y,'fro'));
end