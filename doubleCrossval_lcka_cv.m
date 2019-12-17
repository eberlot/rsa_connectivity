function lCKA_dcv = doubleCrossval_lcka_cv(Data,nPart,nCond)
%function lCKA_dcv = doubleCrossval_lcka_multiReg_cv(Data,nPart,nPart)
% calculates lCKA in different ways across two regions
% only double crossvalidated version
% INPUT:
%       - Data - cell (nReg x 1) - for now only works if nReg = 2
%       - nPart - number of partiitons
%       - nCond - number of conditions

% here calculate Gs for each region
nReg = size(Data,1);
G_all = zeros(nCond*nPart,nCond*nPart,nReg);
for r=1:nReg
    G_all(:,:,r) = Data{r}*Data{r}'/size(Data{r},2);
end
partVec = kron((1:nPart)',ones(nCond,1));
ind = nchoosek(1:nPart,2);
id1 = [ind;fliplr(ind)];
id2 = [ind;ind];
nComb = size(id1,1);
idx=1;
ind1 = zeros(nComb*nComb,2);
ind2 = zeros(nComb*nComb,2);
for p1=1:nComb
    for p2=1:nComb
        ind1(idx,:) = [id1(p1,1) id1(p1,2)];
        ind2(idx,:) = [id2(p2,1) id2(p2,2)];
        idx=idx+1;
    end
end
idx_ccv = ind1(:,1)~=ind1(:,2) & ind2(:,1)~=ind2(:,2) & sum(ind1==ind2,2)~=2 & sum(fliplr(ind1)==ind2,2)~=2; % double crossvalidated
t = [ind1(idx_ccv==1,:) ind2(idx_ccv==1,:)];
t = unique(t,'rows');
ind1 = t(:,1:2);
ind2 = t(:,3:4);
fprintf('\nCalculating connectivity for region:\n');
lcka = zeros(nReg,nReg,size(ind1,1));
for r=1:nReg
    reg1 = r;
    reg2 = ~ismember(1:nReg,r);
    for i=1:size(ind1,1)      
        g1 = G_all(partVec==ind1(i,1),partVec==ind1(i,2),reg1);
        g2 = G_all(partVec==ind2(i,1),partVec==ind2(i,2),reg2);
        tmp1 = rsa_vectorizeIPMfull(g1);
        tmp2 = rsa_vectorizeIPMfull(g2);
        A=(bsxfun(@minus,tmp1,mean(tmp1,2)))'; % zero-mean
        B=squeeze(bsxfun(@minus,tmp2,mean(tmp2,2))); % zero-mean
        A=bsxfun(@times,A,1./sqrt(sum(A.^2,1))); %% L2-normalization
        B=bsxfun(@times,B,1./sqrt(sum(B.^2,1))); %% L2-normalization
        C=sum(bsxfun(@times,A,B)); % correlation
        lcka(reg1,reg2,i)=C;
    end
fprintf('%d.',r);
end
lCKA_dcv.ncv = mean(lcka,3);
