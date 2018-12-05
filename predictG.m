function [T,predG,cor,cosDist]=predictG(G1,G2)
% function [predG,r,cosDist]=predictG(G1,G2)
% predicts G2 from G1
% INPUT:
% - G1: crossvalidated G matrix from reg1
% - G2: crossvalidated G matrix from reg2
%
% OUTPUT:
% - T:      transformation matrix
% - predG:  predicted G matrix
% - cor:    correlation between predG and G2
% - cosDist:cosine distnace between predG and G2

% check if both Gs of the same size
if size(G1)~=size(G2)
    error('G1 and G2 must be of the same size!');
end

% decompose G1 - eigenvalue decomposition
% G1
[V1,L1]     = eig(G1);                     
[l,i]       = sort(diag(L1),1,'descend'); % sort the eigenvalues
V1          = V1(:,i);
U1          = bsxfun(@times,V1,ssqrt(l'));
% G2
[V2,L2]     = eig(G2);                    
[l,i]       = sort(diag(L2),1,'descend');          
V2          = V2(:,i);
U2          = bsxfun(@times,V2,ssqrt(l'));

% transformation matrix T - A*A'
A = pinv(U1)*U2;
T = A*A'; 

predG = U1*T*U1';
% assess the fit
Gtrue   = rsa_vectorizeIPMfull(G2);
Gpred   = rsa_vectorizeIPMfull(predG);
cor     = corr(Gtrue',Gpred');
cosDist = pdist([Gtrue;Gpred],'cosine');

