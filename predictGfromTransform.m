function [predG,varargout]=predictGfromTransform(G1,T,varargin)
% function [predG,r,cosDist]=predictGfromTransform(G1,T,varargin)
% predicts G2 from G1 using T matrix
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
% VARARGOUT:
% - r: correlation between predG and G2
% - cosDist: cosine distance between predG and G2 (if G2 provided)
%
vararginoptions(varargin,{'G2'}); 

% decompose G1 - eigenvalue decomposition
% G1
[V1,L1]     = eig(G1);                     
[l,i]       = sort(diag(L1),1,'descend'); % sort the eigenvalues
V1          = V1(:,i);
U1          = bsxfun(@times,V1,ssqrt(l'));

predG = U1*T*U1';

if exist('G2')
    % assess the fit
    Gtrue   = rsa_vectorizeIPMfull(G2);
    Gpred   = rsa_vectorizeIPMfull(predG);
    cor     = corr(Gtrue',Gpred');
    cosDist = pdist([Gtrue;Gpred],'cosine');
    varargout(1)={cor};
    varargout(2)={cosDist};
end

