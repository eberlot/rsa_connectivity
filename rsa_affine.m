function [T,R,S,rms]=rsa_affine(D1,D2)
% function [T R S]=rsa_affine(D1,D2)
% calculates an affine transformation between two RDMs
% T - translation
% R - rotation
% S - scale
% rms - root mean square error of the deviation

% roughly based on:
% https://stackoverflow.com/questions/13432805/finding-translation-and-scale-on-two-sets-of-points-to-get-least-square-error-in
% EBerlot, Oct 2018

N = size(D1,1);
m = ones(1,N)/N;

if size(D1)~=size(D2)
    fprintf('Provided RDMs must be of the same size');
else
    % calculate the centroids
    C1=mean(D1,2);
    C2=mean(D2,2);
    % subtract centroids
    D1=bsxfun(@minus,D1,C1);
    D2=bsxfun(@minus,D2,C2);
    
    % calculate C - the covariance matrix of the coordinates
	%C = D1*diag(m)*D2'; 
    C = pinv(D1)*D2; 
    % singular value decomposition
    [V,S,W] = svd(C) ;   % singular value decomposition
    I = eye(N) ;
    if (det(V*W') < 0)   % more numerically stable than using (det(C) < 0)
        I(N,N) = -1 ;
    end
    % rotation
    R = W*I*V' ;
    % translation
	T = C2-R*C1 ;

    % calculate deviation - root mean square error
	Diff = S*D1 - D2;    
    % how do you combine here translation and rotation?!
	rms = sqrt(sum(sum(bsxfun(@times,m,Diff).*Diff)));  
end
