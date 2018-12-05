
% mix G3 as G1 and G2 with different proportions
w1 = [0.1:0.1:0.9];

U1=normrnd(0,1,[5,6]);
U2=normrnd(0,1,[5,15]);
G1 = U1*U1';
G2 = U2*U2';
G1 = G1./trace(G1);
G2 = G2./trace(G2);

for f=1:length(w1)
    G3=w1(f)*G1 + (1-w1(f))*G2;
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
    % G3
    [V3,L3]     = eig(G3);
    [l,i]       = sort(diag(L3),1,'descend');
    V3          = V3(:,i);
    U3          = bsxfun(@times,V3,ssqrt(l'));
    
    % transformation matrix A and T - A*A'
    A13 = pinv(U1)*U3;
    A23 = pinv(U2)*U3;
    T13 = A13*A13';
    T23 = A23*A23';
    
    % plot
    figure(f)
    subplot(241)
    imagesc(G1)
    title('G1')
    subplot(242)
    imagesc(G2)
    title('G2')
    subplot(243)
    imagesc(G3)
    title(sprintf('G3 = %1.1f x G1+%1.1f x G2',w1(f),1-w1(f)));
    subplot(245)
    imagesc(A13)
    title('A1-3');
    subplot(246)
    imagesc(A23)
    title('A2-3');
    subplot(247)
    imagesc(T13)
    title('T1-3');
    subplot(248)
    imagesc(T23)
    title('T2-3');
end

keyboard;