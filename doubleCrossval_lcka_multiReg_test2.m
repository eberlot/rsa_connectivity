function lCKA_dcv = doubleCrossval_lcka_multiReg_test2(Data,nPart,nCond)
%function lCKA_dcv = doubleCrossval_lcka_multiReg_test2(Data,nPart,nPart)
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

nReg = size(Data,1);
partVec = kron((1:nPart)',ones(nCond,1));
H = eye(nCond)-ones(nCond)./nCond;
% calculate Gs for all regions
G_all = zeros(nCond*nPart,nCond*nPart,nReg);
for r=1:nReg
    G_all(:,:,r) = Data{r}*Data{r}'/size(Data{r},2);
end
% indices to chose for partition combinations
ind = nchoosek(1:nPart,2);
id1 = [ind;fliplr(ind);[(1:nPart)' (1:nPart)']];
id2 = id1;
for r1=1:nReg
    for r2 = find(~ismember(1:nReg,r1))
        idx1=1;
        % initialize
        G_tmp1 = zeros(nCond,nCond,size(id1,1)*size(id2,1));
        G_tmp2 = zeros(nCond,nCond,size(id1,1)*size(id2,1));
        % test
        G_test1 = G_tmp1;
        G_test2 = G_tmp2;
        % end
        partComb = zeros(size(id1,1)*size(id2,1),4);
        for i1=1:size(id1,1)
            for i2=1:size(id2,1)
                g1 = G_all(partVec==id1(i1,1),partVec==id1(i1,2),r1);
                g2 = G_all(partVec==id2(i2,1),partVec==id2(i2,2),r2);
                G_tmp1(:,:,idx1) = H*g1*H'; % double centered
                G_tmp2(:,:,idx1) = H*g2*H'; % double centered
                partComb(idx1,:)=[id1(i1,1) id1(i1,2) id2(i2,1) id2(i2,2)];
                idx1 = idx1+1;
            end
        end
%         % for ncv - try different code (from previous!)
%         % determine combinations of partitions to use
%         ind = nchoosek(1:nPart,2);
%         id1 = [ind;fliplr(ind);[(1:nPart)' (1:nPart)']];
%         id2 = [ind;ind;[(1:nPart)' (1:nPart)']];
%         nComb = size(id1,1);
%         idx=1;
%         ind1 = zeros(nComb*nComb,2);
%         ind2 = zeros(nComb*nComb,2);
%         for p1=1:nComb
%             for p2=1:nComb
%                 ind1(idx,:) = [id1(p1,1) id1(p1,2)];
%                 ind2(idx,:) = [id2(p2,1) id2(p2,2)];
%                 idx=idx+1;
%             end
%         end
%         t = [ind1 ind2];
%         t = unique(t,'rows');
%         ind1 = t(:,1:2);
%         ind2 = t(:,3:4);
%         idx_ccv_tmp = ind1(:,1)~=ind1(:,2) & ind2(:,1)~=ind2(:,2) & sum(ind1==ind2,2)~=2 & sum(fliplr(ind1)==ind2,2)~=2; % double crossvalidated
%         findCCV = t(idx_ccv_tmp,:);
%         % find corresponding entries in partComb
%         find((sum(diff(sort(partComb,2),1,2)~=0,2)+1)==4) = zeros(size(findCCV,1),1);
%         for i=1:size(findCCV,1)
%             [~,idx_ccv_test(i,:)]=ismember(findCCV(i,:),partComb,'rows');
%         end
%         % end of test
        idx_ncv = [1:(size(partComb,1))]';
        idx_cv  = find(partComb(:,1)~=partComb(:,2) & partComb(:,3)~=partComb(:,4)); % don't use the same run
        % test
        idx_ccv = find((sum(diff(sort(partComb,2),1,2)~=0,2)+1)==4); % all different
        % end of test
        % here try out different ones
        % only don't use always the same run
        idx_test1 = find((sum(diff(sort(partComb,2),1,2)~=0,2)+1)~=1);
        idx_test2 = find(partComb(:,1)~=partComb(:,2) & partComb(:,3)~=partComb(:,4) & sum(partComb(:,1:2)==partComb(:,3:4),2)~=2); %& sum(fliplr(partComb(:,1:2))==partComb(:,3:4)2)~=2
        idx_test3 = find(partComb(:,1)~=partComb(:,2) & partComb(:,3)~=partComb(:,4) & ...
            (sum(partComb(:,1:2)==partComb(:,3:4),2))~=2 & (sum(fliplr(partComb(:,1:2))==partComb(:,3:4),2)~=2));
        % now do different lcka-s
        lCKA_dcv.ncv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_ncv),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_ncv),3)')');
        lCKA_dcv.cv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_cv),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_cv),3)')');
        lCKA_dcv.ccv(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_ccv),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_ccv),3)')');
        %lCKA_dcv.ccv_test(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_ccv_test),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_ccv_test),3)')');
        
        
        lCKA_dcv.test1(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_test1),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_test1),3)')');
        lCKA_dcv.test2(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_test2),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_test2),3)')');
        lCKA_dcv.test3(r1,r2) = corr(rsa_vectorizeIPMfull(nanmean(G_tmp1(:,:,idx_test3),3)')',rsa_vectorizeIPMfull(nanmean(G_tmp2(:,:,idx_test3),3)')');
    end
end


% for comparison
% condVec = kron(ones(nPart,1),(1:nCond)');
% X = indicatorMatrix('identity_p',condVec);
% for l=1:nReg
%     D{l} = pinv(X)*Data{l};
%     G1 = D{l}*D{l}'/size(D{l},2);
%     G{l} = H*G1*H'; % double centered
% end
% c = corr(rsa_vectorizeIPMfull(G{1}')',rsa_vectorizeIPMfull(G{2}')');