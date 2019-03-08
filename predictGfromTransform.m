function [predG,varargout]=predictGfromTransform(G1,T,varargin)
% function [predG,r,cosDist]=predictGfromTransform(G1,T,varargin)
% predicts G2 from G1 using the transformation T matrix
% INPUT:
% - G1: crossvalidated G matrix from reg1
% - T:  transformation matrix (obtained using calcTransformG, or feature matrix)
%
% VARARGIN:
% - G2: true G2 (if provided calculates the prediction - true G fit)
%
% OUTPUT:
% - predG:  predicted G matrix
%
% VARARGOUT (if true G given):
% - r:          correlation between predG and G2
% - corDist:    correlation distance between predG and G2 (if G2 provided)
% - cosDist:    cosine ditance between predG and G2 (if G2 provided)
vararginoptions(varargin,{'G2'}); 

% decompose G1 - eigenvalue decomposition
% G1
[V1,L1]     = eig(G1);                     
[l,i]       = sort(diag(L1),1,'descend'); % sort the eigenvalues
V1          = V1(:,i);
U1          = bsxfun(@times,V1,real(sqrt(l')));

predG = U1*T*U1';

if exist('G2')
    % assess the fit
    Gtrue   = rsa_vectorizeIPMfull(G2);
    Gpred   = rsa_vectorizeIPMfull(predG);
    cor     = corr(Gtrue',Gpred');
    corDist = pdist([Gtrue;Gpred],'correlation');
    cosDist = pdist([Gtrue;Gpred],'cosine');
    varargout(1)={cor};
    varargout(2)={corDist};
    varargout(3)={cosDist};
end

