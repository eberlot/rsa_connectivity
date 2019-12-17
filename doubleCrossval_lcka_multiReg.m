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
ind = nchoosek(1:nPart,2);
id1 = [ind;fliplr(ind);[(1:nPart)' (1:nPart)']];
id2 = [ind;ind;[(1:nPart)' (1:nPart)']];
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
% 
% ind = [ind;[(1:nPart)' (1:nPart)']];
% nComb = size(ind,1);
% % pre-define all combinations
% ind1 = zeros(nComb*(nComb+1)/2,2); ind2=ind1;
% idx=1;
% for p1=1:nComb
%     for p2=1:nComb
%         for v=1:2 % reg 1 treated twice (upper and lower triangular)
%             if v==1
%                 ind1(idx,:) = [ind(p1,1) ind(p1,2)];
%             else
%                 ind1(idx,:) = [ind(p1,2) ind(p1,1)];
%             end
%             ind2(idx,:) = [ind(p2,1) ind(p2,2)];
%             idx=idx+1;
%         end
%     end
% end
% ind1 = [ind1;[(1:nPart)' (1:nPart)']];
% ind2 = [ind2;[(1:nPart)' (1:nPart)']];
% lcka = zeros(nReg,nReg,size(ind1,1));
t = [ind1 ind2];
t = unique(t,'rows');
ind1 = t(:,1:2);
ind2 = t(:,3:4);
% fprintf('\nCalculating connectivity for region:\n');
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
%fprintf('%d.',r);
end
% here indices for crossvalidated version and double crossvalidated
idx_cv = ind1(:,1)~=ind1(:,2) & ind2(:,1)~=ind2(:,2); % crossvalidated
idx_ccv = ind1(:,1)~=ind1(:,2) & ind2(:,1)~=ind2(:,2) & sum(ind1==ind2,2)~=2 & sum(fliplr(ind1)==ind2,2)~=2; % double crossvalidated
lCKA_dcv.ncv = mean(lcka,3);
lCKA_dcv.cv = mean(lcka(:,:,idx_cv),3);
lCKA_dcv.ccv = mean(lcka(:,:,idx_ccv),3);
