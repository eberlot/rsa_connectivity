function lCKA_dcv = doubleCrossval_lcka_multiReg(Data,nPart,nCond)
%function lCKA_dcv = doubleCrossval_lcka_multiReg(Data,nPart,nPart)
% calculates lCKA in different ways across two regions
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
H = eye(nCond)-ones(nCond)./nCond;
ind = nchoosek(1:nPart,2);
ind = [ind;[(1:nPart)' (1:nPart)']];
nComb = size(ind,1);
lcka = zeros(nReg,nReg,nComb*(nComb+1)/2);
ind1 = zeros(nComb*(nComb+1)/2,2); ind2=ind1;
fprintf('\nCalculating connectivity for region:\n');
for r=1:nReg
    idx=1;
    for p1=1:nComb
        for p2=p1:nComb
            reg1 = r;
            reg2 = ~ismember(1:nReg,r);
            g1 = G_all(partVec==ind(p1,1),partVec==ind(p1,2),reg1);
            g2 = G_all(partVec==ind(p2,1),partVec==ind(p2,2),reg2);
            tmp1 = H*g1*H';
            tmp2 = arrayfun(@(x) H*g2(:,:,x)*H',1:size(g2,3),'UniformOutput',false);
            tmp2 = reshape(cell2mat(tmp2),[nCond,nCond,nReg-1]);
            tmp1 = rsa_vectorizeIPMfull(tmp1);
            tmp2 = rsa_vectorizeIPMfull(tmp2); 
            A=(bsxfun(@minus,tmp1,mean(tmp1,2)))'; % zero-mean
            B=squeeze(bsxfun(@minus,tmp2,mean(tmp2,2))); % zero-mean
            A=bsxfun(@times,A,1./sqrt(sum(A.^2,1))); %% L2-normalization
            B=bsxfun(@times,B,1./sqrt(sum(B.^2,1))); %% L2-normalization
            C=sum(bsxfun(@times,A,B)); % correlation
            lcka(reg1,reg2,idx)=C;
            if r==1
                ind1(idx,:) = [ind(p1,1) ind(p1,2)];
                ind2(idx,:) = [ind(p2,1) ind(p2,2)];
            end
            idx = idx+1;
        end
    end
    fprintf('%d.',r);
end
% here indices for crossvalidated version and double crossvalidated
idx_cv = ind1(:,1)~=ind1(:,2) & ind2(:,1)~=ind2(:,2); % crossvalidated
idx_ccv = ind1(:,1)~=ind1(:,2) & ind2(:,1)~=ind2(:,2) & sum(ind1==ind2,2)~=2; % double crossvalidated
lCKA_dcv.ncv = mean(lcka,3);
lCKA_dcv.cv = mean(lcka(:,:,idx_cv),3);
lCKA_dcv.ccv = mean(lcka(:,:,idx_ccv),3);
