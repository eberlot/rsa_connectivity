% combinations for double crossvalidation
nPart = 8;
% level 1
H1 = indicatorMatrix('allpairs',1:nPart); % first level
C1_ncv = nPart*(nPart+1)/2;        % / size(H1,1)+nPart - number of all combinations for level 1
C1_cv  = nPart*(nPart-1)/2;        % / size(H1,1) - number of combinations for level 1 crossval  

% level2
H2 = indicatorMatrix('allpairs',1:size(H1,1));
C2_ncv = (C1_ncv*(C1_ncv+1))/2;  % all possible combinations - with no crossvalidation ever
C2_cv  = C1_cv*(C1_cv-1)/2;      % / size(H2,1)

% conservative choice - no overlapping partition
comb = nchoosek(1:nPart,4);
comb = [comb;fliplr(comb)];