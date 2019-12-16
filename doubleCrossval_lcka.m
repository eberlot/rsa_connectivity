function lCKA_dcv = doubleCrossval_lcka(Data,nPart,nCond)
%function lCKA_dcv = doubleCrossval_lcka(Data,nPart,nPart)
% calculates lCKA in different ways across two regions
% INPUT:
%       - Data - cell (nReg x 1)
%       - nPart - number of partiitons
%       - nCond - number of conditions

partVec = kron((1:nPart)',ones(nCond,1));
% here calculate Gs
H = eye(nCond)-ones(nCond)./nCond;
ind = nchoosek(1:nPart,2);
ind = [ind;[(1:nPart)' (1:nPart)']];
idx=1;
for p1=1:length(ind)
    for p2=p1:length(ind)
        tmp1 = Data{1}(partVec == ind(p1,1),:)*Data{1}(partVec == ind(p1,2),:)'/size(Data{1},2);
        tmp2 = Data{2}(partVec == ind(p2,1),:)*Data{2}(partVec == ind(p2,2),:)'/size(Data{2},2);
        tmp1 = H*tmp1*H';
        tmp2 = H*tmp2*H';
        tmp1 = rsa_vectorizeIPMfull(tmp1);
        tmp2 = rsa_vectorizeIPMfull(tmp2);
        T.lcka(idx,1) = corr(tmp1',tmp2');
        T.ind1(idx,:) = [ind(p1,1) ind(p1,2)]; 
        T.ind2(idx,:) = [ind(p2,1) ind(p2,2)];
        idx = idx+1;
    end
end
% here indices for crossvalidated version and double crossvalidated
idx_cv = T.ind1(:,1)~=T.ind1(:,2) & T.ind2(:,1)~=T.ind2(:,2); % crossvalidated
idx_ccv = T.ind1(:,1)~=T.ind1(:,2) & T.ind2(:,1)~=T.ind2(:,2) & sum(T.ind1==T.ind2,2)~=2; % double crossvalidated
lCKA_dcv.ncv = mean(T.lcka);
lCKA_dcv.cv = mean(T.lcka(idx_cv));
lCKA_dcv.ccv = mean(T.lcka(idx_ccv));
