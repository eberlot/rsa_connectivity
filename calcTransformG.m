function [T,predG,cor,corDist,cosDist]=calcTransformG(G1,G2)
% function [T,predG,r,corDist,cosDist]=calcTransformG(G1,G2)
% calculates transformation matrix T that translates G1 to G2
% INPUT:
% - G1: crossvalidated G matrix from reg1
% - G2: crossvalidated G matrix from reg2
%
% OUTPUT:
% - T:          transformation matrix
% - predG:      predicted G matrix
% - cor:        correlation between predG and G2
% - corDist:    correlation distance between predG and G2
% - cosDist:    cosine distance between predG and G2

% check if both Gs of the same size
if size(G1)~=size(G2)
    error('G1 and G2 must be of the same size!');
end

% decompose G1 - eigenvalue decomposition
% G1
[V1,L1]     = eig(G1);                     
[l,i]       = sort(diag(L1),1,'descend'); % sort the eigenvalues
V1          = V1(:,i);
U1          = bsxfun(@times,V1,real(sqrt(l')));
% G2
[V2,L2]     = eig(G2);                    
[l,i]       = sort(diag(L2),1,'descend');          
V2          = V2(:,i);
U2          = bsxfun(@times,V2,real(sqrt(l')));

% transformation matrix T - A*A'
A = pinv(U1)*U2;
T = A*A'; 
predG = U1*T*U1';
% assess the fit
Gtrue   = rsa_vectorizeIPMfull(G2);
Gpred   = rsa_vectorizeIPMfull(predG);
cor     = corr(Gtrue',Gpred'); 
corDist = pdist([Gtrue;Gpred],'correlation');
cosDist = pdist([Gtrue;Gpred],'cosine');

