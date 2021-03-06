function lCKA_dcv = doubleCrossval_lcka_multiReg_test4(Data,nPart,nCond)
%function lCKA_dcv = doubleCrossval_lcka_multiReg_test4(Data,nPart,nPart)
% calculates lCKA in different ways across two regions
% INPUT:
%       - Data - cell (nReg x 1) 
%       - nPart - number of partititons
%       - nCond - number of conditions
% OUTPUT:
%       - structure lCKA_dcv with fields
%           * ncv - not crossvalidated version
%           * cv  - crossvalidated (on the level of regions)
%           * dcv - double crossvalidated (within and across regions)
%

% here calculate Gs for each region
nReg = size(Data,1);
G_all = zeros(nCond*nPart,nCond*nPart,nReg);
for r=1:nReg
    G_all(:,:,r) = (Data{r}*Data{r}')/size(Data{r},2);
end
partVec = kron((1:nPart)',ones(nCond,1));
H = eye(nCond)-ones(nCond)./nCond;
ind = nchoosek(1:nPart,2);
id1 = [ind;fliplr(ind);[(1:nPart)' (1:nPart)']];
id2 = id1;
for r1=1:nReg
    for r2 = find(~ismember(1:nReg,r1))
        idx1=1;
        % initialize
        G_tmp1 = zeros(nCond,nCond,size(id1,1)*size(id2,1));
        G_tmp2 = zeros(nCond,nCond,size(id1,1)*size(id2,1));
        G_test1 = G_tmp1;
        G_test2 = G_tmp2;
        partComb = zeros(size(id1,1)*size(id2,1),4);
        for i1=1:size(id1,1)
            for i2=1:size(id2,1)
                g1 = (Data{r1}(partVec==id1(i1,1),:)*Data{r1}(partVec==id1(i1,2),:)')/size(Data{r1},2);
                g2 = (Data{r2}(partVec==id1(i2,1),:)*Data{r2}(partVec==id2(i2,2),:)')/size(Data{r2},2);
                %g1 = G_all(partVec==id1(i1,1),partVec==id1(i1,2),r1);
                %g2 = G_all(partVec==id2(i2,1),partVec==id2(i2,2),r2);
                G_tmp1(:,:,idx1) = H*g1*H'; % double centered
                G_tmp2(:,:,idx1) = H*g2*H'; % double centered
                cross_test(:,idx1) = times(rsa_vectorizeIPMfull(G_tmp1(:,:,idx1))',rsa_vectorizeIPMfull(G_tmp2(:,:,idx1))');
                g_test(:,:,idx1) = g1;
                corr_tmp(idx1) = corr(rsa_vectorizeIPMfull(G_tmp1(:,:,idx1))',rsa_vectorizeIPMfull(G_tmp2(:,:,idx1))');
                % now new version here too
                tmp1 = rsa_vectorizeIPMfull(g1);
                tmp2 = rsa_vectorizeIPMfull(g2);
                A=(bsxfun(@minus,tmp1,mean(tmp1,2)))'; % zero-mean
                B=squeeze(bsxfun(@minus,tmp2,mean(tmp2,2)))'; % zero-mean
                A=bsxfun(@times,A,1./sqrt(sum(A.^2,1))); %% L2-normalization
                B=bsxfun(@times,B,1./sqrt(sum(B.^2,1))); %% L2-normalization
                G_test1(:,:,idx1) = rsa_squareIPMfull(A');
                G_test2(:,:,idx1) = rsa_squareIPMfull(B');
                corr_test(idx1) = sum(bsxfun(@times,A,B));
                partComb(idx1,:)=[id1(i1,1) id1(i1,2) id2(i2,1) id2(i2,2)];
                idx1 = idx1+1;
            end
        end
        
        % test
        for i1=1:size(id1,1)
            g_tmp(:,:,i1) = (Data{r}(partVec==id1(i1,1),:)*Data{r}(partVec==id1(i1,2),:)')/size(Data{r},2);
            g_tmp2(:,:,i1) = G_all(partVec==id1(i1,1),partVec==id1(i1,2),r1);
        end
        % end of test
        
        
        
        idx_ncv = [1:(size(partComb,1))]';
        idx_cv  = find(partComb(:,1)~=partComb(:,2) & partComb(:,3)~=partComb(:,4)); % don't use the same run
        %idx_ccv = find((sum(diff(sort(partComb,2),1,2)~=0,2)+1)==4); % all different
        idx_ccv = find(partComb(:,1)~=partComb(:,2) & partComb(:,3)~=partComb(:,4) & ...
            (sum(partComb(:,1:2)==partComb(:,3:4),2))~=2 & (sum(fliplr(partComb(:,1:2))==partComb(:,3:4),2)~=2));
        
        
        lCKA_dcv.tmp_ncv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_ncv),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_ncv),3)')');
        lCKA_dcv.tmp_cv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_cv),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_cv),3)')');
        lCKA_dcv.tmp_ccv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_ccv),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_ccv),3)')');
        
        lCKA_dcv.test_ncv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_test1(:,:,idx_ncv),3)')',rsa_vectorizeIPMfull(nanmean(G_test2(:,:,idx_ncv),3)')');
        lCKA_dcv.test_cv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_test1(:,:,idx_cv),3)')',rsa_vectorizeIPMfull(nanmean(G_test2(:,:,idx_cv),3)')');
        lCKA_dcv.test_ccv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_test1(:,:,idx_ccv),3)')',rsa_vectorizeIPMfull(nanmean(G_test2(:,:,idx_ccv),3)')');
        
        lCKA_dcv.direct_tmp_ncv(r1,r2) = mean(corr_tmp(idx_ncv));
        lCKA_dcv.direct_tmp_cv(r1,r2) = mean(corr_tmp(idx_cv));
        lCKA_dcv.direct_tmp_ccv(r1,r2) = mean(corr_tmp(idx_ccv));
        
        lCKA_dcv.direct_test_ncv(r1,r2) = mean(corr_test(idx_ncv));
        lCKA_dcv.direct_test_cv(r1,r2) = mean(corr_test(idx_cv));
        lCKA_dcv.direct_test_ccv(r1,r2) = mean(corr_test(idx_ccv));
        
    end
end
% for comparison
condVec = kron(ones(nPart,1),(1:nCond)');
X = indicatorMatrix('identity_p',condVec);
for l=1:nReg
    D{l} = pinv(X)*Data{l};
    G1 = (D{l}*D{l}')/size(D{l},2);
    G_test{l} = H*G1*H'; % double centered
end
c = corr(rsa_vectorizeIPMfull(G_test{1})',rsa_vectorizeIPMfull(G_test{2})');
crossP = times(rsa_vectorizeIPMfull(G_test{1})',rsa_vectorizeIPMfull(G_test{2})');

