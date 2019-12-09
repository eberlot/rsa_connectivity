function lCKA_dcv = doubleCrossval_lcka(Data,nPart,nCond)
%function lCKA_dcv = doubleCrossval_lcka(Data,nPart,nPart)
% calculates lCKA in different ways across two regions
% INPUT:
%       - Data - cell (nReg x 1)
%       - nPart - number of partiitons
%       - nCond - number of conditions

partVec = kron((1:nPart)',ones(nCond,1));
condVec = kron(ones(nPart,1),(1:nCond)');
% here calculate Gs
H = eye(nCond)-ones(nCond)./nCond;
T1 = Data{1}*Data{1}'/size(Data{1},2);
T2 = Data{2}*Data{2}'/size(Data{2},2);
% version 1 - everything
ind = nchoosek(1:nPart,2);
ind = [ind;[(1:nPart)' (1:nPart)']];
R_ncv = []; TNC1 = []; TNC2 = [];
idx=1;
for p1=1:length(ind)
    for p2=p1:length(ind)
        tmp1 = T1(partVec == ind(p1,1), partVec == ind(p1,2));
        tmp2 = T2(partVec == ind(p2,1), partVec == ind(p2,2));
        tmp1 = H*tmp1*H';
        tmp2 = H*tmp2*H';
        tmp1 = rsa_vectorizeIPM(tmp1);
        tmp2 = rsa_vectorizeIPM(tmp2);
        tmp_corr = corr(tmp1',tmp2');
        R_ncv = [R_ncv,tmp_corr];
        TNC1 = [TNC1;tmp1'];
        TNC2 = [TNC2;tmp2'];
      %  fprintf('%d:\t%d-%d:\t comparing %d-%d with %d-%d\n',idx,p1,p2,ind(p1,1),ind(p1,2),ind(p2,1),ind(p2,2))
        idx = idx+1;
    end
end

% version 2 - crossvalidated (but also overlapping)

ind = indicatorMatrix('allpairs',[1:nPart]);
R_cv = []; TC1 = []; TC2 = [];
idx=1;
for p1=1:size(ind,1)
    for p2=p1:size(ind,1) 
        tmp1 = T1(partVec == find(ind(p1,:)==1), partVec == find(ind(p1,:)==-1));
        tmp2 = T2(partVec == find(ind(p2,:)==1), partVec == find(ind(p2,:)==-1));
        tmp1 = H*tmp1*H';
        tmp2 = H*tmp2*H';
        tmp1 = rsa_vectorizeIPM(tmp1);
        tmp2 = rsa_vectorizeIPM(tmp2);
        tmp_corr = corr(tmp1',tmp2');
        R_cv = [R_cv,tmp_corr];
        TC1 = [TC1;tmp1'];
        TC2 = [TC2;tmp2'];
       % fprintf('%d:\t%d-%d:\t comparing %d-%d with %d-%d\n',idx,p1,p2,find(ind(p1,:)==1),find(ind(p1,:)==-1),find(ind(p2,:)==1),find(ind(p2,:)==-1))
        idx = idx+1;
    end
end

% version 3 - non-overlapping run combination
ind = indicatorMatrix('allpairs',[1:nPart]);
R_ccv = []; TCC1 = []; TCC2 = [];
idx=1;
for p1=1:size(ind,1)
    for p2=p1:size(ind,1)
        if p1~=p2
            tmp1 = T1(partVec == find(ind(p1,:)==1), partVec == find(ind(p1,:)==-1));
            tmp2 = T2(partVec == find(ind(p2,:)==1), partVec == find(ind(p2,:)==-1));
            tmp1 = H*tmp1*H';
            tmp2 = H*tmp2*H';
            tmp1 = rsa_vectorizeIPM(tmp1);
            tmp2 = rsa_vectorizeIPM(tmp2);
            tmp_corr = corr(tmp1',tmp2');
            R_ccv = [R_ccv,tmp_corr];
            TCC1 = [TCC1;tmp1'];
            TCC2 = [TCC2;tmp2'];
           % fprintf('%d:\t%d-%d:\t comparing %d-%d with %d-%d\n',idx,p1,p2,find(ind(p1,:)==1),find(ind(p1,:)==-1),find(ind(p2,:)==1),find(ind(p2,:)==-1))
            idx = idx+1;
        end
    end
end

lCKA_dcv.ncv = mean(R_ncv);
lCKA_dcv.cv = mean(R_cv);
lCKA_dcv.ccv = mean(R_ccv);