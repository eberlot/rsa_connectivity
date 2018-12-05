U1=normrnd(0,1,[5,6]);
U2 = U1;
G1 = U1*U1';
%G1 = G1 + eye(size(G1,1));
G2 = U2*U2';
%G2 = G2 + eye(size(G2,1));
%G1 = pcm_makePD(G1);
%G2 = pcm_makePD(G2);
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

% transformation matrix A and T - A*A'
A12 = pinv(U1)*U2;
A21 = pinv(U2)*U1;
T12 = A12*A12';
T21 = A21*A21';
%T12 = pinv(U1)*G2*pinv(U1)';
%T21 = pinv(U2)*G1*pinv(U2)';


predG2 = U1*T12*U1';


