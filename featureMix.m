% feature transformation
U1=normrnd(0,1,[5,6]);
U2 = U1;
G1 = U1*U1';
G2 = U2*U2';

[U1,l1]=eig(G1);
[U2,l2]=eig(G2);
[l1,i1]=sort(diag(l1),1,'descend');
[l2,i2]=sort(diag(l2),1,'descend');
U1=U1(:,i1);
U2=U2(:,i2);
U1=bsxfun(@times,U1,sqrt(l1')); 
U2=bsxfun(@times,U2,sqrt(l2')); 

% feature as first eigenvector
t1 = U1(:,1);
t2 = U2(:,1);
F1 = t1*t1';
F2 = t2*t2';
F1 = round(F1,3);
F2 = round(F2,3);
G1_red = F1*F1';
G2_red = F2*F2';

A12 = pinv(U1)*U2; 
A21 = pinv(U2)*U1; 

T12 = pinv(U1)*G2*pinv(U1)'; 
T21 = pinv(U2)*G1*pinv(U2)'; 

omega12 = pinv(F1)*G2_red*pinv(F1)';
omega21 = pinv(F2)*G1_red*pinv(F2)';


G2_redpred = F1*omega12*F1';