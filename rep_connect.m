function varargout = rep_connect(what,varargin)
%function varargout = rep_connect(what,varargin)
%calculates connectivity between layers of DNN (resnet50, vgg16)

%baseDir = '/Volumes/MotorControl/data/rsa_connectivity/noise_injection';
baseDir = '/Volumes/MotorControl/data/rsa_connectivity/new_normalization';
codeDir = '~/Documents/MATLAB/Projects/rsa_connectivity';
aType = {'correlation','cosine'};
dType = {'univariate','multi-squared','multi-sqrt'};
style.file(fullfile(codeDir,'DNN_style.m'));
style.use('default');
%load(fullfile(baseDir,'Comb'));
switch what
    case 'firstLevel:wholePattern'
        % use this if estimating first level on whole patterns
        % otherwise submit the act to calculate function directly
        DNNname = 'alexnet'; % alexnet, resnet50, vgg16, alexnet_imagenet
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        act=DNN_connect('HOUSEKEEPING:normalizeUnits',act);   
        [U,RDM,RDM_sqrt,~,~,G,~]=DNN_connect('firstLevel:calculate',act,size(act{1},1));
        save(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname)),'U','RDM','RDM_sqrt','G');
        varargout={U,RDM,RDM_sqrt,G};
    case 'firstLevel:calculate'
        % calculate first level estimates: U, RDM, G (cRDM, cG)
        Data = varargin{1};
        nCond = varargin{2};
        % initialize
        nPart = size(Data{1},1)/nCond;
        numLayer = size(Data,1);
        partVec = kron((1:nPart)',ones(nCond,1));
        condVec = kron(ones(nPart,1),(1:nCond)');
        G = cell(numLayer,1);
        cG = G;
        % contrast matrix for G->distances
        C = indicatorMatrix('allpairs',1:nCond);
        X = indicatorMatrix('identity_p',condVec);
        H = eye(nCond)-ones(nCond)./nCond;
        RDM = zeros(numLayer,size(C,1));
        cRDM = RDM; RDM_sqrt = RDM; cRDM_sqrt = RDM; % crossval, square root
        U = zeros(numLayer,nCond);
     %   fprintf('First level calculating layer:\n');
        for i=1:numLayer
       %     fprintf('%d.',i);
            nVox = size(Data{i},2);
            t=Data{i};
            for j=1:nPart % first remove the mean of each run
                t(partVec==j,:)=bsxfun(@minus,Data{i}(partVec==j,:),mean(Data{i}(partVec==j,:),1));
            end
            D               = pinv(X)*Data{i};
            G{i}            = D*D'/nVox;
            G{i}            = H*G{i}*H';
            RDM(i,:)        = diag(C*G{i}*C')';
            RDM_sqrt(i,:)   = ssqrt(RDM(i,:));
            D               = pinv(X)*t;
            U(i,:)          = mean(D,2)';
            cG{i}           = pcm_estGCrossval(Data{i},partVec,condVec);
            cG{i}           = cG{i}./trace(cG{i});
            cRDM(i,:)       = diag(C*cG{i}*C')';
            cRDM_sqrt(i,:)  = ssqrt(cRDM(i,:));
        end
    %    fprintf('...done all.\n');
        varargout{1}=U;
        varargout{2}=RDM;
        varargout{3}=RDM_sqrt;
        varargout{4}=cRDM;
        varargout{5}=cRDM_sqrt;
        varargout{6}=G;
        varargout{7}=cG;
    case 'firstLevel:normalize_overall'
        % transform activation units to z-scores
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        act_z = act;
        for i=1:nLayer
            a=act{i};
            [Y,I] = sort(a(:)); 
           % q = Y./length(Y) - 1/(2*length(Y));
            q = (1:length(Y))./length(Y) - 1/(2*length(Y));
            z = norminv(q,0,1);
            act_z{i} = reshape(z(I),size(a)); % reorder & reshape
        end
        act = act_z;
        dirZ = fullfile(baseDir,sprintf('%s_zscore',DNNname));
        dircheck(dirZ);
        save(fullfile(dirZ,sprintf('%s_zscore_activations',DNNname)),'act');
    case 'firstLevel:normalize_perUnit'
        % normalize activation units (per unit responses to stimuli)
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        for i=1:nLayer
            for j=1:size(act{i},2)
                [Y,~] = sort(act{i}(:,j));
                % q = Y./length(Y) - 1/(2*length(Y));
                q = (1:length(Y))./length(Y) - 1/(2*length(Y));
                z = norminv(q,0,1);
                act{i}(:,j) = z;
            end
            fprintf('Done: layer-%d\n',i);
        end
        dirZ = fullfile(baseDir,sprintf('%s_normUnit',DNNname));
        dircheck(dirZ);
        save(fullfile(dirZ,sprintf('%s_normUnit_activations',DNNname)),'act');
    case 'firstLevel:positive'
        % only keep the positive weights, set the rest to 0
        % mimics the rectified linear unit in the networks
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        nLayer = size(act,1);
        for i=1:nLayer
            a=act{i};
            a(a<=0)=0;
            act{i}=a;
        end
        dirP = fullfile(baseDir,sprintf('%s_positive',DNNname));
        dircheck(dirP);
        save(fullfile(dirP,sprintf('%s_positive_activations',DNNname)),'act','-v7.3');
    case 'firstLevel:plot_act'
        % plot activation for each layer - per unit res
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in act
        act=DNN_connect('HOUSEKEEPING:normalizeUnits',act);   
        nLayer = size(act,1);
        nFig = ceil(nLayer/8); % how many figures needed
        
        idx = 0; % count number of layers
        for i=1:nFig
            idxF = 1; % for figure
            figure
            for j=1:8 % arbitrary number of layers plotted
                subplot(2,4,idxF)
                histogram(act{idx+1}(:));
                title(sprintf('layer-%d',idx+1));
                idxF = idxF+1;
                idx=idx+1;
            end
        end
    case 'firstLevel:plot_dist'
        % plot distances for each layer (squared or square root)
        DNNname = 'alexnet';
        metric = 'RDM'; % RDM or RDM_sqrt
        vararginoptions(varargin,{'DNNname','metric'});
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname))); % load in act
        nLayer = size(RDM,1);
        nFig = ceil(nLayer/8); % how many figures needed
        
        idx = 0; % count number of layers
        D = eval(metric);
        for i=1:nFig
            idxF = 1; % for figure
            for j=1:8 % arbitrary number of layers plotted
                figure(5)
                subplot(2,4,idxF)
                histogram(D(idx+1,:));
                title(sprintf('layer-%d',idx+1));
                figure(10*5)
                subplot(2,4,idxF);
                imagesc(rsa_squareRDM(D(idx+1,:))); colorbar;
                title(sprintf('layer-%d',idx+1));
                idxF = idxF+1;
                idx=idx+1;
            end
        end
    case 'firstLevel:hist_dist'
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname))); % load in act
       
        figure
        subplot(131)
        scatterplot(RDM(1,:)',RDM(2,:)'); xlabel('layer-1'); ylabel('layer-2');
        subplot(132)
        scatterplot(RDM(1,:)',RDM(3,:)'); xlabel('layer-1'); ylabel('layer-3');
        subplot(133)
        scatterplot(RDM(2,:)',RDM(3,:)'); xlabel('layer-2'); ylabel('layer-3');
        
    case 'secondLevel:calcConnect'
        % gets the second level metrics
        % second level: univariate / RDM connect, transformation of Gs
        % saves the outputs
        DNNname = 'vgg16';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_firstLevel',DNNname)));
        % estimate second level metrics (between RDMs, Gs)
        % 1) calculate distance based on mean activity (univariate)
        alpha{1} = DNN_connect('secondLevel:estimate_distance',U,'univariate');
        % 2) calculate distance between RDMs (cosine, correlation)
        alpha{2} = DNN_connect('secondLevel:estimate_distance',RDM,'RDM'); 
        % 3) calculate distance between RDMs (cosine, correlation) - sqrt
        alpha{3} = DNN_connect('secondLevel:estimate_distance',RDM_sqrt,'RDM'); 
        % 4) calculate transformation matrices T between Gs
        alpha{4} = DNN_connect('secondLevel:transformG',G);
        varargout{1}=alpha;
        % save the variables
        save(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
    case 'secondLevel:estimate_distance' 
        % esimate distance between layers - 2nd level
        RDM = varargin{1};
        dataType = varargin{2}; % univariate or multivariate connectivity
        alpha=[];
        for i=1:2
            D = DNN_connect('secondLevel:calcDist',RDM,aType{i});
            D.distType = ones(size(D.l1))*i;
            D.distName = repmat(aType(i),size(D.l1));
            alpha = addstruct(alpha,D);
        end
        varargout{1}=alpha;
        fprintf('Done estimating undirected distance metrics: %s.\n',dataType); 
    case 'secondLevel:calcDist'
            % calculate correlation / cosine distances
            rdm = varargin{1};
            distType = varargin{2};
            % calculate distance metric from the input
            % input: N x D matrix (N - number of RDMs; D - distance pairs)
            % dist types: 'correlation' or 'cosine'
            % output: structure D, with fields:
            %   - D: pairwise distances of RDMs
            %   - l1: indicator which rdm / layer taken as first
            %   - l2: indicator which rdm ; layer taken as second
            switch (distType)
                case 'correlation'
                    % additional step for correlation - first remove the mean
                    rdm  = bsxfun(@minus,rdm,mean(rdm,2));
            end
            nRDM = size(rdm,1);
            rdm  = normalizeX(rdm);
            tmpR  = rdm*rdm'; % correlation across RDMs
            %D.dist = 1-rsa_vectorizeRDM(tmpR)'; % distances - as
            %dissimilarity
            D.dist = rsa_vectorizeRDM(tmpR)'; % distances - as similarity
            ind=indicatorMatrix('allpairs',(1:nRDM));
            % determine each element of the pair
            [~,D.l1]=find(ind==1);
            i=ismember(ind,-1);
            D.l2 = sum(cumprod(i==0,2),2)+1;
            varargout{1}=D;     
    case 'secondLevel:transformG'
        % here estimate T + derive all metrics
        regularize=0;
        G=varargin{1};
        vararginoptions(varargin(2:end),{'regularize'});
        numG = max([size(G,1) size(G,2)]);
        if regularize
            for i=1:numG
                G{i}=G{i}+eye(size(G{i}));
            end
        end
        TT=[];
        for i1=1:numG % all pairs (because T is not symmetric)
            for i2=1:numG
                T.l1=i1;
                T.l2=i2;
                [Tr,~,~,~]              = calcTransformG(G{T.l1},G{T.l2}); % can retrieve cosine distance
                T.T                     = rsa_vectorizeIPMfull(Tr);
                T.distType              = ones(size(T.l1));
                % here characterise T
                % scaling
                scalar                  = mean(diag(Tr)); % which scalar
                scaleTr                 = eye(size(Tr))*scalar; % simplified transform - only scalar
                [~,~,T.scaleDist]       = predictGfromTransform(G{T.l1},scaleTr,'G2',G{T.l2});
                % diagonal + rangei1
                diagTr                  = Tr.*eye(size(Tr));
                [~,~,T.diagDist]        = predictGfromTransform(G{T.l1},diagTr,'G2',G{T.l2});
                T.diagRange             = max(diag(Tr))-min(diag(Tr));
                T.diagStd               = std(diag(Tr));
                % eigenvalues
                [~,T.eigRange,T.eigStd,T.eigComp,~] = eig_complexity(Tr);
                TT=addstruct(TT,T);
            end
        end
        varargout{1}=TT;
    case 'secondLevel:plotRDM'
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname))); % load in act
        idx=1;
        numFig = randi(99);
        for i=1:3
            figure
            if i==1
                anchorCols = [0 0 1; 1 1 1; 1 0 0];
                cols = colorScale(anchorCols,256,1);
            else
                betCol = [245 132 132]./255;
                anchorCols = [1 1 1; betCol; 1 0 0];
                cols = colorScale(anchorCols,256,1);
            end
            figure(numFig);
            a=alpha{i};
            subplot(3,2,idx)
            minimum = min(a.dist(a.distType==1));
            maximum = max(a.dist(a.distType==1));
            if i==1
                v = max([abs(minimum) maximum]);
                minimum = -v;
                maximum = v;
            end
            %limit = max([abs(minimum) abs(maximum)]);
            imagesc(rsa_squareRDM(a.dist(a.distType==1)'),[minimum maximum]); title(sprintf('%s - corr-RDM',dType{i})); 
            colormap(gca,cols); colorbar;
            subplot(3,2,idx+1)
            minimum = min(a.dist(a.distType==2));
            maximum = max(a.dist(a.distType==2));
            if i==1
                v = max([abs(minimum) maximum]);
                minimum = -v;
                maximum = v;
            end
            %limit = max([abs(minimum) abs(maximum)]);
            imagesc(rsa_squareRDM(a.dist(a.distType==2)'),[minimum maximum]); title(sprintf('%s - cos-RDM',dType{i})); 
            idx=idx+2;
            colormap(gca,cols); colorbar;
        end
      %  colormap hot;
    case 'isomap:undirected'
        n_dim   = 2; % number of dimensions to consider
        n_neigh = 2; % number of neighbours to consider
        DNNname = 'alexnet';
        vararginoptions(varargin,{'n_dim','n_neigh','DNNname'});

        a = load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
        A = a.alpha;
        numLayer = length(unique([A{1}.l1;A{1}.l2]));
        mColor = cell(1,numLayer);
        mCol = hsv(numLayer);
        for i=1:numLayer
            mColor{i}=mCol(i,:);
        end
        trueOrder = squareform(pdist((1:numLayer)'));
        for d = 1:3 % univariate or multivariate-squared, multi-sqrt
            figure
            idx=1;
            for m = 1:2 % cosine or correlation
                D = rsa_squareRDM(A{d}.dist(A{d}.distType==m)');
                [mX,mp] = topology_estimate(D,n_dim,n_neigh); % submit to topology function
                subplot(2,2,idx)
                imagesc(D); colormap hot;
                title(sprintf('%s - %s, RDM',dType{d},aType{m}));
                %tauAll = corr(rsa_vectorizeRDM(D)',rsa_vectorizeRDM(trueOrder)','Type','Kendall');
                %title(sprintf('%s - %s, tau: %0.3f',dType{d},aType{m},tauAll));
                subplot(2,2,idx+1)
                hold on;
                W = full(mp.D);
                [r,c,val] = find(W);
                val = val./max(val); % renormalize
                for i=1:length(r)
                  %  plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',(1/val(i)),'Color',repmat(val(i),3,1)./(max(val)+0.1));
                    plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',1,'Color',repmat(val(i),3,1)./(max(val)+0.1));
                end
               scatterplot(mX(:,1),mX(:,2),'label',(1:numLayer),'split',(1:numLayer)','markercolor',mColor,'markertype','.','markersize',30); 
               axis equal; axis off;
               title(sprintf('%s - %s, isomap',dType{d},aType{m}));
               idx=idx+2;
            end
        end
    case 'isomap:directed'
        n_dim   = 3; % number of dimensions to consider
        n_neigh = 3; % number of neighbours to consider
        DNNname = 'vgg16';
        vararginoptions(varargin,{'n_dim','n_neigh','DNNname'});
        a = load(fullfile(baseDir,DNNname,sprintf('%s_alpha',DNNname)),'alpha');
        A = a.alpha{4};
        metrics={'scaleDist','diagDist','eigComp'};
        
        numLayer = length(unique([A.l1;A.l2]));
        mColor = cell(1,numLayer);
        mCol = hsv(numLayer);
        for i=1:numLayer
            mColor{i}=mCol(i,:);
        end
        trueOrder = squareform(pdist((1:numLayer)'));
        figure
        idx=1;
        for m = 1:length(metrics)
            t = rsa_squareIPMfull(A.(metrics{m})'); % t+t' to make it undirected
            D = t+t'; % symmetrize
            [mX,mp] = topology_estimate(t+t',n_dim,n_neigh);
            subplot(length(metrics),2,idx)
            imagesc(D); colormap hot;
            %tauAll = corr(rsa_vectorizeRDM(D)',rsa_vectorizeRDM(trueOrder)','Type','Kendall');
           % title(sprintf('%s, tau: %0.3f',metrics{m},tauAll));
           title(sprintf('%s, RDM',metrics{m})); 
           subplot(length(metrics),2,idx+1)
            hold on;
            W = full(mp.D);
            [r,c,val] = find(W);
            val = val./max(val); % renormalize
            for i=1:length(r)
                plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',1,'Color',repmat(val(i),3,1)./(max(val)+0.1));
            end
            scatterplot(mX(:,1),mX(:,2),'label',(1:numLayer),'split',(1:numLayer)','markercolor',mColor,'markertype','.','markersize',30);
            title(sprintf('%s, isomap',metrics{m}));
            idx=idx+2;
        end
     
    case 'doubleCrossval' % old - TO DO
        % combine level 1 + 2
        % estimates per region and across
        % smart double crossvalidation (with non-overlapping folds)
        Data    = varargin{1};
        nPart   = varargin{2};
        nCond   = varargin{3};
        % new variables
        nLayer = size(Data,1);
        partVec = kron((1:nPart)',ones(nCond,1));
        condVec = kron(ones(nPart,1),(1:nCond)');
        nVox = size(Data{1},2);
        H = eye(nCond)-ones(nCond)./nCond;
        C = indicatorMatrix('allpairs',1:nCond);
        X = indicatorMatrix('identity_p',condVec(partVec<3));
        % remove mean from each layer / run
        for i=1:nLayer
            for j=1:nPart % first remove the mean of each run
                Data{i}(partVec==j,:)=bsxfun(@minus,Data{i}(partVec==j,:),mean(Data{i}(partVec==j,:),1));
            end
        end
        % for second level
        layPair = indicatorMatrix('allpairs',1:nLayer);
        Conf = cell(2,1);
        A = cell(2,1);
        for i=1:size(layPair,1)
            reg1 = find(layPair(i,:)==1);
            reg2 = find(layPair(i,:)==-1);
            a_conn  = zeros(size(Comb,1),2);
            for c=1:size(Comb,1);
                % here estimate first level
                c1 = Comb(c,:);
                % first mean activity
                Data1A = pinv(X)*Data{reg1}(ismember(partVec,[c1(1) c1(2)]),:);
                Data1B = pinv(X)*Data{reg1}(ismember(partVec,[c1(3) c1(4)]),:);
                Data2A = pinv(X)*Data{reg2}(ismember(partVec,[c1(5) c1(6)]),:);
                Data2B = pinv(X)*Data{reg2}(ismember(partVec,[c1(7) c1(8)]),:);
                % second G / RDM
                G1 = Data1A*Data1B'/nVox;
                G2 = Data2A*Data2B'/nVox;
                G1 = H*G1*H'; % double center
                G2 = H*G2*H';
                RDM(1,:) = diag(C*G1*C')'; % squared
                RDM(2,:) = diag(C*G2*C')';
                % third alpha
                for cc=1:2 % corr/cos
                    if cc==1 % corr - extra mean subtraction
                        rdm = bsxfun(@minus,RDM,mean(RDM,2));
                    else
                        rdm = RDM;
                    end
                    rdm   = normalizeX(rdm);
                    tmpR  = rdm*rdm'; % correlation across RDMs
                    a_conn(c,cc) = 1-rsa_vectorizeRDM(tmpR)'; % distances
                end
            end
            for m = 1:2 % for now only 1:2
                Conf{m}(reg1,reg2) = var(a_conn(:,m));
                Conf{m}(reg2,reg1) = var(a_conn(:,m));
                A{m}(reg1,reg2)    = mean(a_conn(:,m));
                A{m}(reg2,reg1)    = mean(a_conn(:,m));
            end
           % fprintf('Done %d/%d.\n',i,size(layPair,1));
        end
        varargout{1}=A; % connectivity matrix
        varargout{2}=Conf; % confidence matrix
    case 'noiseless:constructTruth_allUnits'
        % here plot the obtained connectivity for all metrics with no noise
        % using all units here
        nPart       = 8;
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nUnits','DNNname'});
        
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in activations
        nCond = size(act{1},1);
        nLayer = size(act,1);
        C=pcm_indicatorMatrix('allpairs',1:nCond);  % Contrast vector for all pairs
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        % here subsample + repeat the true pattern for each partition
        data = cell(nLayer,1);
        for i=1:nLayer
            rUnits = randperm(size(act{i},2)); % randomise the order of units
            data{i} = repmat(act{i},nPart,1);
        end
        % here establish the true distance structure (TD) - noiseless
        [tD{1},tD{2},~,tD{3}]=rep_connect('firstLevel:calculate',data,nCond);
        TD{1} = rep_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
        TD{2} = rep_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
        TD{3} = rep_connect('secondLevel:calcDist',tD{3},aType{1}); % cRDM-squared-corr
        TD{4} = rep_connect('secondLevel:calcDist',tD{3},aType{2}); % cRDM-squared-cos
        [TD{7},~] = calcCKA(data,nPart,size(act{1},1),'average');       % lCKA
        % weighted covariances
        varD = rsa_varianceLDC(zeros(nCond),C,eye(nCond),nPart,500); % Get the variance
        TD{8}   = cosineW(tD{3},varD);
        plotDist = [1:4,7];
        trueOrder = squareform(pdist((1:nLayer)'));
        for i=1:5    
            if i<5
                Dist = squareform(TD{plotDist(i)}.dist');
            else
                Dist = TD{plotDist(i)};
            end
            minimum = min(Dist(:));
            maximum = max(Dist(:));
            if i==1
                anchorCols = [0 0 1; 1 1 1; 1 0 0];
                cols = colorScale(anchorCols,256,1);
                v = max([abs(minimum) maximum]);
                minimum = -v;
                maximum = v;
            else
                betCol = [245 132 132]./255;
                anchorCols = [1 1 1; betCol; 1 0 0];
                cols = colorScale(anchorCols,256,1);
            end
            figure(100)
            subplot(2,3,i)
            imagesc(Dist);
            colormap(gca,cols); colorbar;
            colorbar;
            hold on;
            [true_accuOrder,~,true_misplaced] = compareOrder(trueOrder,Dist);
            title(sprintf('Accuracy, %1.2f, %d misplaced',true_accuOrder,true_misplaced));
        end
    case 'noiseless:constructTruth_subsets'
        % construct connectivity with no noise, subsets of units
        nPart       = 8;
        nUnits      = 500;
        nSimulation = 100; % how many simulations to perform
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nUnits','DNNname','nSimulation'});
        
        VV = [];
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in activations
        nCond = size(act{1},1);
        nLayer = size(act,1);
        trueOrder = squareform(pdist((1:nLayer)'));
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        % here subsample + repeat the true pattern for each partition
        fprintf('Running simulations:\n');
        for s=1:nSimulation
            act_subsets = cell(nLayer,1);
            data = cell(nLayer,1);
            for i=1:nLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                data{i} = repmat(act_subsets{i},nPart,1);
                %data{i} = repmat(act{i},nPart,1);
            end
            % here establish the true distance structure (TD) - noiseless
            [tD{1},tD{2},~,tD{3}]=rep_connect('firstLevel:calculate',data,nCond);
            TD{1} = rep_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
            TD{2} = rep_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
            TD{3} = rep_connect('secondLevel:calcDist',tD{3},aType{1}); % cRDM-squared-corr
            TD{4} = rep_connect('secondLevel:calcDist',tD{3},aType{2}); % cRDM-squared-cos
            [TD{5},TD{6}] = anzellottiDist(data,nPart,size(act{1},1));  % Anzellotti
            [TD{7},~] = calcCKA(data,nPart,size(act{1},1),'average');       % lCKA
            for i=1:7
                if i<5
                    Dist = squareform(TD{i}.dist');
                else
                    Dist = TD{i};
                end
                [V.true_accuOrder,~,V.true_misplaced] = compareOrder(trueOrder,Dist);
                % other info
                V.connMatrix = rsa_vectorizeRDM(Dist);
                V.metric = i;
                V.nSim = s;
                VV = addstruct(VV,V);
            end
            fprintf('%d.',s);
        end
        keyboard;
        % here plot the average across simulations
        plotDist = [1:5,7];
        for i=1:6 %5    
            Dist = squareform(mean(VV.connMatrix(VV.metric==i,:),1));
            minimum = min(Dist(:));
            maximum = max(Dist(:));
            if i==1
                anchorCols = [0 0 1; 1 1 1; 1 0 0];
                cols = colorScale(anchorCols,256,1);
                v = max([abs(minimum) maximum]);
                minimum = -v;
                maximum = v;
            else
                betCol = [245 132 132]./255;
                anchorCols = [1 1 1; betCol; 1 0 0];
                cols = colorScale(anchorCols,256,1);
            end
            figure(99)
            subplot(2,3,i)
            imagesc(Dist);
            colormap(gca,cols); colorbar;
            colorbar;
            hold on;
            title(sprintf('Accuracy, %1.2f, %1.1f misplaced',mean(VV.true_accuOrder(VV.metric==plotDist(i))),...
                mean(VV.true_accuOrder(VV.true_misplaced==plotDist(i)))));
        end
    case 'noise:oneNoisy_plot'
        % here plot the connectivity when one region is more noisy
        nPart       = 8;
        nUnits      = 500;
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nUnits','DNNname','nSimulation'});
        
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname))); % load in activations
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        % here subsample + repeat the true pattern for each partition
        fprintf('Running simulations:\n');
        act_subsets = cell(nLayer,1);
        data = cell(nLayer,1);
        for i=1:nLayer
            rUnits = randperm(size(act{i},2)); % randomise the order of units
            act_subsets{i} = act{i}(:,rUnits(1:nUnits));
            data{i} = repmat(act_subsets{i},nPart,1);
            %data{i} = repmat(act{i},nPart,1);
        end
        % here add noise (to one region)
        [Data,~] = addSharedNoise(data,3,0,'within_oneNoisy');
        % calculate metrics
        [tD{1},tD{2},~,tD{3}]=rep_connect('firstLevel:calculate',Data,nCond);
        TD{1} = rep_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
        TD{2} = rep_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
        TD{3} = rep_connect('secondLevel:calcDist',tD{3},aType{1}); % cRDM-squared-corr
        TD{4} = rep_connect('secondLevel:calcDist',tD{3},aType{2}); % cRDM-squared-cos
        [TD{5},TD{6}] = anzellottiDist(Data,nPart,size(act{1},1));  % Anzellotti
        [TD{7},~] = calcCKA(Data,nPart,size(act{1},1),'average');   % lCKA

        keyboard;
        % here plot the connectivity matrices
        plotDist = [1:5,7];
        for i=1:6 %5    
            if i<5
                Dist = squareform(TD{plotDist(i)}.dist');
            else
                Dist = TD{plotDist(i)};
            end
            minimum = min(Dist(:));
            maximum = max(Dist(:));
            if i==1
                anchorCols = [0 0 1; 1 1 1; 1 0 0];
                cols = colorScale(anchorCols,256,1);
                v = max([abs(minimum) maximum]);
                minimum = -v;
                maximum = v;
            else
                betCol = [245 132 132]./255;
                anchorCols = [1 1 1; betCol; 1 0 0];
                cols = colorScale(anchorCols,256,1);
            end
            figure(99)
            subplot(2,3,i)
            imagesc(Dist);
            colormap(gca,cols); colorbar;
            colorbar;
            hold on;
        end
        
    case 'noise:simulate_crossval' % old - TO RUN
        % here simulate with different crossvalidation options
        % combine levels 1 and 2
        % for now no G-metric
        nPart       = 8;
        nSim        = 10;
        nUnits      = 500;
        noiseType   = 'shared_harmful'; 
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nPart','nSim','noiseType','dataType','DNNname'});
        % initialize
        % note: 
        % dataInd:
        % 1-uni; 2-RDMsquared; 3-RDMsqrt; 4-cRDMsquared; 5-cRDMsqrt;
        % 6-dcRDMsquared; 7-dcRDMsqrt; 8-PCs
        % alphaInd
        % 1 - corr, 2 - cos, 3 - Anzellotti
        dataInd     = [1 2 2 3 3 4 4 5 5 6 6 7 7 8]; % anzelotti as 8
        alphaInd    = [1 1 2 1 2 1 2 1 2 1 2 1 2 3];
        RDMInd      = [1 3 4 5 6 3 4 5 6 3 4 5 6 0];
        % load and initialise
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        trueOrder = squareform(pdist((1:nLayer)'));
        switch noiseType
            case 'shared_helpful' % true order
                varReg = 0:0.25:2;
                corrReg = 0:0.2:0.8;
            case 'shared_harmful' % shuffled order
                % shuffle the order of activation units
                [act,trueOrder] = DNN_connect('HOUSEKEEPING:shuffled_structure',act);
                varReg = 0.5:0.5:2;
                corrReg = [0,0.2,0.4,0.8];
        end
        VV=[]; CC=[];
        for v=varReg        % within noise
            if v==0
                corR=0;
            else
                corR=corrReg;
            end
            for r=corR         % correlated noise 
                for n=1:nSim    % number of simulations
                    % here subsample + repeat the true pattern for each partition
                    act_subsets = cell(nLayer,1);
                    data = cell(nLayer,1);
                    for i=1:nLayer
                        rUnits = randperm(size(act{i},2)); % randomise the order of units
                        act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                        data{i} = repmat(act_subsets{i},nPart,1);
                    end
                    tElapsed=tic;
                    Data = addSharedNoise(data,v,r,noiseType);
                    [RDMconsist,RDMconsist_all] = rdmConsist(Data,nPart,size(act_subsets{1},1));
                    [fD{1},fD{2},fD{3},fD{4},fD{5},~,~]=DNN_connect('firstLevel:calculate',Data,nCond); 
                    [W_dcv,Conf_dcv] = DNN_connect('doubleCrossval',Data,nPart,size(act_subsets{1},1));
                    % order: uni, RDM, RDM_sqrt, cRDM, cRDM_sqrt
                    % cycle around all combinations of data / metrics
                    for i=1:length(dataInd)
                        if dataInd(i) < 6
                            t = DNN_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                            T = rsa_squareRDM(t.dist');
                            V.corrNoiseless = corr(TD.dist(RDMInd(i),:)',rsa_vectorizeRDM(T)');
                        elseif dataInd(i) > 5 && dataInd(i) < 8 % dcRDM-squared
                            T = W_dcv{(dataInd(i)-6)*2+alphaInd(i)};
                            V.corrNoiseless = corr(TD.dist(RDMInd(i),:)',rsa_vectorizeRDM(T)');
                            C.conf = rsa_vectorizeRDM(Conf_dcv{(dataInd(i)-6)*2+alphaInd(i)});
                            C.conn = rsa_vectorizeRDM(T);
                            C.dataType = dataInd(i);
                            C.metricType = alphaInd(i);
                            C.numSim = n;
                            C.corrReg = r;
                            C.varReg = v;
                            CC = addstruct(CC,C);
                        else % Anzellotti
                            T = anzellottiDist(Data,nPart,size(act{1},1));       
                            V.corrNoiseless = NaN;
                        end
                        NN              = construct_neighbourhood(T);
                        V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                        V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(T)');  % from original distances
                        V.RDM           = rsa_vectorizeRDM(T);
                        V.RDMconsist    = RDMconsist;           
                        V.RDMconsist_all= RDMconsist_all;
                        V.dataType      = dataInd(i); % uni / multi / directional
                        V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                        V.metricIndex   = i;
                        V.numSim        = n;
                        V.corrReg       = r;
                        V.varReg        = v;
                        VV=addstruct(VV,V);
                    end
                    fprintf('%d. ',n);
                    toc(tElapsed);
                end
                fprintf('\nFinished variance %d.\n\n',v);
            end
            fprintf('\nFinished correlation %d.\n\n',r);
        end
        save(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s_doubleCrossval',noiseType)),'-struct','VV');
        save(fullfile(baseDir,DNNname,sprintf('simulations_confidence_noise_%s_doubleCrossval',noiseType)),'-struct','CC');
        fprintf('\nDone simulations %s: %s \n',DNNname,noiseType);
    case 'noise:simulate'
        nPart       = 8;
        nSim        = 1000;
        nUnits      = 500;
        noiseType   = 'within'; % allEqual or neighbours
        DNNname     = 'alexnet';
        varReg = [0:.25:4,5:1:10];
        corrReg = 0;
        vararginoptions(varargin,{'nPart','nSim','noiseType','dataType','DNNname','varReg','corrReg'});
        % initialize
        % 1) corr with uni; 2) corr with RDM-squared; 3) corr with cRDM-squared;
        % 4) cos with cRDM-squared; 5) Anzellotti-R2 6) Anzellotti-r
        % 7) lCKA; 8) new covariance weighting
        dataInd     = [1 2 3 3 4 5 6 7]; % anzelotti as 5-6, 7-lCKA, 8-cosineW
       % RDMInd      = [1 2 3 3]; % just for uni-RDM-cRDM
        alphaInd    = [1 1 1 2 3 3 4 5];
        % crossval and non-crossval treated as the same
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        nCond = size(act{1},1);
        nLayer = size(act,1);
        C=pcm_indicatorMatrix('allpairs',1:nCond);  % Contrast vector for all pairs 
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        trueOrder = squareform(pdist((1:nLayer)'));
        VV=[]; 
        for n=1:nSim    % number of simulations
            fprintf('\nSimulation %d:\n',n);
            tElapsed=tic;
            % here subsample + repeat the true pattern for each partition
            act_subsets = cell(nLayer,1);
            if strcmp(noiseType,'shared_harmful')
              %  [act,trueOrder,orderShuff] = rep_connect('HOUSEKEEPING:shuffled_structure',act); % shuffle the order
            else
                spatialOrder = trueOrder;
            end
            data = cell(nLayer,1);
            for i=1:nLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                data{i} = repmat(act_subsets{i},nPart,1);
            end
            % here establish the true distance structure (TD)
            % when still noiseless
            [tD{1},tD{2},~,tD{3}]=rep_connect('firstLevel:calculate',data,nCond);
            TD{1} = rep_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
            TD{2} = rep_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
            TD{3} = rep_connect('secondLevel:calcDist',tD{3},aType{1}); % cRDM-squared-corr
            TD{4} = rep_connect('secondLevel:calcDist',tD{3},aType{2}); % cRDM-squared-cos
            [TD{5},TD{6}] = anzellottiDist(data,nPart,size(act{1},1));  % Anzellotti
            [TD{7},~] = calcCKA(data,nPart,size(act{1},1),'average');       % lCKA
            % weighted covariances
            varD = rsa_varianceLDC(zeros(nCond),C,eye(nCond),nPart,nUnits); % Get the variance
            TD{8}   = cosineW(tD{3},varD);
            for v=varReg        % within noise
                for r=corrReg
                    if v==0
                        Data = data;
                        if strcmp(noiseType,'shared_harmful')
                            spatialOrder = squareform(pdist(randperm(8)'));
                        end
                    else
                        if strcmp(noiseType,'shared_harmful')
                            [Data,spatialOrder] = addSharedNoise(data,v,r,noiseType);
                            spatialOrder = squareform(pdist(spatialOrder'));
                        else
                            Data = addSharedNoise(data,v,r,noiseType);
                        end
                    end
                    [fD{1},fD{2},~,fD{3},~,~,~]=rep_connect('firstLevel:calculate',Data,nCond);
                    Data = rep_connect('HOUSEKEEPING:removeRunMean',Data,nPart,nCond);
                    [RDMconsist,~,Conf_corr,Conf_cos] = rdmConsist(Data,nPart,size(act_subsets{1},1));
                    % cycle around all combinations of data / metrics
                    for i=1:length(dataInd)
                        if i < 5
                            t = rep_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                            T = rsa_squareRDM(t.dist');
                            trueD = rsa_squareRDM(TD{i}.dist');
                        elseif i==5  % Anzellotti, Rs
                            T = anzellottiDist(Data,nPart,size(act{1},1));
                            trueD = TD{i};
                        elseif i==6 % Anzellotti, r
                            [~,T] = anzellottiDist(Data,nPart,size(act{1},1));
                            trueD = TD{i};
                        else
                            [T,~] = calcCKA(Data,nPart,size(act{1},1),'average'); 
                           trueD = TD{i}; 
                        end
                        V.corrNoiseless = corr(rsa_vectorizeRDM(trueD)',rsa_vectorizeRDM(T)');
                        % add other info
                        [V.true_accuOrder,V.true_countMiss,V.true_misplaced] = compareOrder(trueOrder,T);  
                        [V.spatial_accuOrder,V.spatial_countMiss,V.spatial_misplaced] = compareOrder(spatialOrder,T);
                        NN              = construct_neighbourhood(T);
                        V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                        V.tauSpatial_NN = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(NN)'); % correlation with spatial noise
                        V.RDM           = rsa_vectorizeRDM(T);
                        V.RDMconsist    = RDMconsist;
                        V.conf_corr     = Conf_corr; % confidence estimates on connectivity
                        V.conf_cos      = Conf_cos;
                        V.dataType      = dataInd(i); % uni / multi / directional
                        V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                        V.metricIndex   = i;
                        V.numSim        = n;
                        V.corrReg       = r;
                        V.varReg        = v;
                        VV=addstruct(VV,V);
                    end     % data
                    fprintf('- correlation: %d.\n',r);
                end
                fprintf('Variance: %d.\n',v);
            end % variance
            toc(tElapsed);
        end
        save(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)),'-struct','VV'); %1: 1:7:0.2:3.4 2:0:0.25:3
        % for now - v1 with normalization, v2 without
        fprintf('\nDone within simulations %s.\n\n',DNNname);
    case 'noise:simulate_shared'  % OLD
        % include only metrics that will be used for OHBM
        nPart       = 8;
        nSim        = 50;
        nUnits      = 500;
        noiseType   = 'shared_harmful';
        DNNname     = 'alexnet';
        varReg = 0:0.25:3;
        corrReg = [0,0.1,0.2,0.4];
        vararginoptions(varargin,{'nPart','nSim','noiseType','dataType','DNNname','corrReg','varReg'});
        % initialize
        % 1) corr with uni; 2) corr with RDM-squared; 3) corr with RDM-sqrt;
        % 4) corr with cRDM-squared; 5) cos with cRDM-squared;
        % 6) corr with cRDM-sqrt; 7) cos with cRDM-sqrt;
        % 8) Anzellotti
        dataInd     = [1 2 3 4 4 5 5 6]; % anzelotti as 8
        alphaInd    = [1 1 1 1 2 1 2 3];
        RDMInd      = [1 2 3 2 4 3 5 6]; % which RDM to compare with for 'truth' noiseless
        % crossval and non-crossval treated as the same
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        spatialOrder = squareform(pdist((1:nLayer)')); % spatial structure (and how noise is imposed)  
        
        VV=[];
        if exist(fullfile(baseDir,DNNname,'simulations_noise_shared'),'file')
            fprintf('This simulation of %s shared already exists - skipping...\n',DNNname);
        else
            for n=1:nSim    % number of simulations
                fprintf('\nSimulation %d:\n',n);
                tElapsed=tic;
                % here subsample + repeat the true pattern for each partition
                act_subsets = cell(nLayer,1);
                data = cell(nLayer,1);
                [act,trueOrder] = DNN_connect('HOUSEKEEPING:shuffled_structure',act); % shuffle the order
                % true order after shuffling
                for i=1:nLayer
                    rUnits = randperm(size(act{i},2)); % randomise the order of units
                    act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                    data{i} = repmat(act_subsets{i},nPart,1);
                end
                % here establish the true distance structure (TD)
                % when still noiseless
                [tD{1},tD{2},tD{3},~,~,~,~]=DNN_connect('firstLevel:calculate',data,nCond);
                TD{1} = DNN_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
                TD{2} = DNN_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
                TD{3} = DNN_connect('secondLevel:calcDist',tD{3},aType{1}); % RDM-sqrt-corr
                TD{4} = DNN_connect('secondLevel:calcDist',tD{2},aType{2}); % RDM-squared-cos
                TD{5} = DNN_connect('secondLevel:calcDist',tD{3},aType{2}); % RDM-sqrt-cos
                TD{6} = anzellottiDist(data,nPart,size(act{1},1));       % Anzellotti
                for v=varReg        % within noise
                    if v==0
                        corR=0;
                    else
                        corR=corrReg;
                    end
                    for r=corR         % correlated noise
                        Data = addSharedNoise(data,v,r,noiseType);
                        for nreg = 1:size(Data,1)
                            Data2{nreg,:} = Data{nreg}(1:size(Data{nreg},1)/2,:);
                        end
                        [RDMconsist,~,Conf_corr,Conf_cos] = rdmConsist(Data,nPart,size(act_subsets{1},1));
                        % consistency of corr / cosine estimation
                        [fD{1},fD{2},fD{3},fD{4},fD{5},~,~]=DNN_connect('firstLevel:calculate',Data2,nCond);
                        % order: uni, RDM, RDM_sqrt, cRDM, cRDM_sqrt
                        % cycle around all combinations of data / metrics
                        for i=1:length(dataInd)
                            if dataInd(i) < 6
                                t = DNN_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                                T = rsa_squareRDM(t.dist');
                                trueD = rsa_squareRDM(TD{RDMInd(i)}.dist');
                            else % Anzellotti
                                T = anzellottiDist(Data,nPart,size(act{1},1));
                                trueD = TD{RDMInd(i)};
                            end
                            V.corrNoiseless = corr(rsa_vectorizeRDM(trueD)',rsa_vectorizeRDM(T)');
                            % add other info
                            NN              = construct_neighbourhood(T);
                            V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                            V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(T)');  % from original distances
                            V.tauSpatial_NN = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(NN)'); % correlation with spatial noise
                            V.tauSpatial_all = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(T)');
                            V.RDM           = rsa_vectorizeRDM(T);
                            V.RDMconsist    = RDMconsist;
                            V.conf_corr     = Conf_corr; % confidence estimates on connectivity
                            V.conf_cos      = Conf_cos;
                            V.dataType      = dataInd(i); % uni / multi / directional
                            V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                            V.RDMtype       = RDMInd(i);
                            V.metricIndex   = i;
                            V.numSim        = n;
                            V.corrReg       = r;
                            V.varReg        = v;
                            VV=addstruct(VV,V);
                        end     % data
                    end % correlation
                    fprintf('- variance: %d.\n',v);
                end % variance
                toc(tElapsed);
            end
            save(fullfile(baseDir,DNNname,'simulations_noise_shared'),'-struct','VV');
            fprintf('\nDone within simulations %s: %s \n\n',DNNname);
        end
    case 'noise:simulate_shared_helpful_short' % OLD
        %spatial noise corresponds to the true order
        nPart       = 8;
        nSim        = 50;
        nUnits      = 500;
        noiseType   = 'shared_harmful';
        DNNname     = 'alexnet';
        varReg = [0:0.25:2.75];
        corrReg = [0,0.1,0.2,0.4];
        vararginoptions(varargin,{'nPart','nSim','noiseType','dataType','DNNname','corrReg','varReg'});
        % initialize
        % 1) corr with uni; 2) corr with RDM-squared; 3) corr with RDM-sqrt;
        % 4) corr with cRDM-squared; 5) cos with cRDM-squared;
        % 6) corr with cRDM-sqrt; 7) cos with cRDM-sqrt;
        % 8) Anzellotti
        dataInd     = [1 2 3 4 4 5 5 6]; % anzelotti as 8
        alphaInd    = [1 1 1 1 2 1 2 3];
        RDMInd      = [1 2 3 2 4 3 5 6]; % which RDM to compare with for 'truth' noiseless
        % crossval and non-crossval treated as the same
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        spatialOrder = squareform(pdist((1:nLayer)')); % spatial structure (and how noise is imposed)  
        trueOrder = spatialOrder; % the two correspond - noise helpful in recovering the roder
        VV=[];
        
        for n=1:nSim    % number of simulations
            fprintf('\nSimulation %d:\n',n);
            tElapsed=tic;   
            % here subsample + repeat the true pattern for each partition
            act_subsets = cell(nLayer,1);
            data = cell(nLayer,1);
            % no shuffling of order here
            for i=1:nLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                data{i} = repmat(act_subsets{i},nPart,1);
            end
            % here establish the true distance structure (TD)
            % when still noiseless
            [tD{1},tD{2},tD{3},~,~,~,~]=DNN_connect('firstLevel:calculate',data,nCond);
            TD{1} = DNN_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
            TD{2} = DNN_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
            TD{3} = DNN_connect('secondLevel:calcDist',tD{3},aType{1}); % RDM-sqrt-corr
            TD{4} = DNN_connect('secondLevel:calcDist',tD{2},aType{2}); % RDM-squared-cos
            TD{5} = DNN_connect('secondLevel:calcDist',tD{3},aType{2}); % RDM-sqrt-cos
            TD{6} = anzellottiDist(data,nPart,size(act{1},1));       % Anzellotti        
            for v=varReg        % within noise
                if v==0
                    corR=0;
                else
                    corR=corrReg;
                end
                for r=corR         % correlated noise
                    Data = addSharedNoise(data,v,r,noiseType);
                    [RDMconsist,~,Conf_corr,Conf_cos] = rdmConsist(Data,nPart,size(act_subsets{1},1));
                    % consistency of corr / cosine estimation
                    [fD{1},fD{2},fD{3},fD{4},fD{5},~,~]=DNN_connect('firstLevel:calculate',Data,nCond);
                    % order: uni, RDM, RDM_sqrt, cRDM, cRDM_sqrt
                    % cycle around all combinations of data / metrics
                    for i=1:length(dataInd)
                        if dataInd(i) < 6
                            t = DNN_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                            T = rsa_squareRDM(t.dist');
                            trueD = rsa_squareRDM(TD{RDMInd(i)}.dist');
                        else % Anzellotti
                            T = anzellottiDist(Data,nPart,size(act{1},1));    
                            trueD = TD{RDMInd(i)};
                        end
                        V.corrNoiseless = corr(rsa_vectorizeRDM(trueD)',rsa_vectorizeRDM(T)');
                        % add other info
                        NN              = construct_neighbourhood(T);
                        V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                        V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(T)');  % from original distances
                        V.tauSpatial_NN = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(NN)'); % correlation with spatial noise
                        V.tauSpatial_all = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(T)');
                        V.RDM           = rsa_vectorizeRDM(T);
                        V.RDMconsist    = RDMconsist;
                        V.conf_corr     = Conf_corr; % confidence estimates on connectivity
                        V.conf_cos      = Conf_cos;
                        V.dataType      = dataInd(i); % uni / multi / directional
                        V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                        V.RDMtype       = RDMInd(i);
                        V.metricIndex   = i;
                        V.numSim        = n;
                        V.corrReg       = r;
                        V.varReg        = v;
                        VV=addstruct(VV,V);
                    end     % data
                end % correlation
                fprintf('- variance: %d.\n',v);
            end % variance
            toc(tElapsed);
        end
        save(fullfile(baseDir,DNNname,'simulations_noise_shared_helpful_OHBM'),'-struct','VV');
        fprintf('\nDone within simulations %s: %s \n\n',DNNname);
    case 'noise:simulate_shared_confidence' % OLD
        % include only metrics that will be used for OHBM
        nPart       = 8;
        nSim        = 50;
        nUnits      = 500;
        noiseType   = 'shared_harmful'; 
        DNNname     = 'alexnet';
        varReg = [0,0.5:0.5:2.5];
        corrReg = [0,0.1,0.2,0.4,0.8];
        vararginoptions(varargin,{'nPart','nSim','noiseType','dataType','DNNname','corrReg','varReg'});
        % initialize
        % here no sqrt
        % 1) corr with uni; 2) corr with RDM-squared;
        % 3) corr with cRDM-squared; 4) cos with cRDM-squared;
        % 5) corr with double cross; 6) cos with double cross
        % 7) Anzellotti
        dataInd     = [1 2 3 3 4 4 5]; % anzelotti as 8
        alphaInd    = [1 1 1 2 1 2 3];
        RDMInd      = [1 2 2 3 2 3 4]; % which RDM to compare with for 'truth' noiseless
        % crossval and non-crossval treated as the same
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        spatialOrder = squareform(pdist((1:nLayer)')); % spatial structure (and how noise is imposed)  
       % varReg = [0,1.7:0.5:3];
       % corrReg = [0,0.2,0.4,0.8];
        VV=[]; CC=[];
        
        for n=1:nSim    % number of simulations
            tElapsed=tic;
            fprintf('\nSimulation %d:\n',n);
            % here subsample + repeat the true pattern for each partition
            act_subsets = cell(nLayer,1);
            data = cell(nLayer,1);
            [act,trueOrder] = DNN_connect('HOUSEKEEPING:shuffled_structure',act); % shuffle the order
            % true order after shuffling
            for i=1:nLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                data{i} = repmat(act_subsets{i},nPart,1);
            end
            % here establish the true distance structure (TD)
            % when still noiseless
            [tD{1},tD{2},~,~,~,~,~]=DNN_connect('firstLevel:calculate',data,nCond);
            TD{1} = DNN_connect('secondLevel:calcDist',tD{1},aType{1}); % uni-corr
            TD{2} = DNN_connect('secondLevel:calcDist',tD{2},aType{1}); % RDM-squared-corr
            TD{3} = DNN_connect('secondLevel:calcDist',tD{2},aType{2}); % RDM-squared-cos
            TD{4} = anzellottiDist(data,nPart,size(act{1},1));       % Anzellotti        
            for v=varReg        % within noise
                if v==0
                    corR=0;
                else
                    corR=corrReg;
                end
                for r=corR         % correlated noise
                    Data = addSharedNoise(data,v,r,noiseType);
                    for nreg = 1:size(Data,1)
                        Data2{nreg,:} = Data{nreg}(1:size(Data{nreg},1)/2,:);
                    end % use only half of the data (to equate what is used for double crossval
                    [RDMconsist,~] = rdmConsist(Data,nPart,size(act_subsets{1},1));
                    [fD{1},fD{2},~,fD{3},~,~,~]=DNN_connect('firstLevel:calculate',Data2,nCond);
                    [W_dcv,Conf_dcv] = DNN_connect('doubleLevel_old',Data,nPart,size(act_subsets{1},1));
                    %[W_dcv2,Conf_dcv2] = DNN_connect('doubleCrossval',Data,nPart,size(act_subsets{1},1));
                    % order: uni, RDM, RDM_sqrt (not), cRDM, cRDM_sqrt (not)
                    % cycle around all combinations of data / metrics
                    for i=1:length(dataInd)
                        if dataInd(i) < 4
                            t = DNN_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                            T = rsa_squareRDM(t.dist');
                            trueD = rsa_squareRDM(TD{RDMInd(i)}.dist');
                        elseif dataInd(i) == 4
                            T = W_dcv{(dataInd(i)-4)+alphaInd(i)};
                            trueD = rsa_squareRDM(TD{RDMInd(i)}.dist');
                            C.conf = rsa_vectorizeRDM(Conf_dcv{(dataInd(i)-4)+alphaInd(i)});
                            C.conn = rsa_vectorizeRDM(T);
                            C.dataType = dataInd(i);
                            C.metricType = alphaInd(i);
                            C.numSim = n;
                            C.corrReg = r;
                            C.varReg = v;
                            CC = addstruct(CC,C);
                        else % Anzellotti
                            T = anzellottiDist(Data,nPart,size(act{1},1));    
                            trueD = TD{RDMInd(i)};
                        end
                        V.corrNoiseless = corr(rsa_vectorizeRDM(trueD)',rsa_vectorizeRDM(T)');
                        % add other info
                        NN              = construct_neighbourhood(T);
                        V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                        V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(T)');  % from original distances
                        V.tauSpatial_NN = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(NN)'); % correlation with spatial noise
                        V.tauSpatial_all = corr(rsa_vectorizeRDM(spatialOrder)',rsa_vectorizeRDM(T)');
                        V.RDM           = rsa_vectorizeRDM(T);
                        V.RDMconsist    = RDMconsist;
                        V.dataType      = dataInd(i); % uni / multi / directional
                        V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                        V.RDMtype       = RDMInd(i);
                        V.metricIndex   = i;
                        V.numSim        = n;
                        V.corrReg       = r;
                        V.varReg        = v;
                        VV=addstruct(VV,V);
                    end     % data
                end % correlation
                fprintf('- variance: %d.\n',v);
            end % variance
            toc(tElapsed);
        end
        save(fullfile(baseDir,DNNname,'simulations_noise_shared_doubleCross_oldVersion'),'-struct','VV');
        save(fullfile(baseDir,DNNname,'simulations_noise_confidence_doubleCross_oldVersion'),'-struct','CC');
        fprintf('\nDone shared confidence simulations %s: %s \n\n',DNNname);
   
    case 'noise:plot'
        DNNname     = 'alexnet';
        noiseType = 'within_oneNoisy'; % within, within_oneNoisy
        vararginoptions(varargin,{'DNNname','noiseType'});
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType))); 
        T = getrow(T,T.metricIndex<8);
        style.use('FiveColours');
        figure
        subplot(221)
        plt.line(T.varReg,T.true_accuOrder,'split',T.metricIndex);
        subplot(222)
        plt.line(T.varReg,T.corrNoiseless,'split',T.metricIndex);
        subplot(223)
        plt.line(T.varReg,T.true_countMiss,'split',T.metricIndex);
        subplot(224)
        plt.line(T.varReg,T.true_misplaced,'split',T.metricIndex);
        
        t = getrow(T,ismember(T.metricIndex,[1,2,4,5,7]));
        figure
        subplot(131)
        plt.line(t.varReg,t.true_accuOrder,'split',t.metricIndex,'leg',{'uni-corr','RDM-corr','cRDM-cos','MVPD','cosineW'}); ylabel('True order');
        xlabel('Noise');
        subplot(132)
        plt.line(t.varReg,t.corrNoiseless,'split',t.metricIndex); ylabel('Noiseless corr');
        subplot(133)
        plt.line(t.varReg,t.true_misplaced,'split',t.metricIndex); ylabel('Positions missed');
        
        figure
        style.use('FiveColours');
        plt.line(t.varReg,t.RDMconsist);
        hold on;
        style.use('FourColor');
        plt.line(t.varReg,max(t.RDMconsist_all,[],2));
        drawline(0,'dir','horz');
        xlabel('Noise');ylabel('RDM consistency'); title('pattern consistency');
        % add lines at 1.9, 2.5, 2.7 (based on fMRI dataset)
    case 'noise:plot_shared'
        DNNname     = 'alexnet';
        noiseType = 'shared_harmful'; % within, within_oneNoisy
        vararginoptions(varargin,{'DNNname','noiseType'});
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType))); 
        T = getrow(T,T.metricIndex<8);
        style.use('FourColours');
        corrR = unique(T.corrReg)';
        for i=corrR
            t = getrow(T,ismember(T.metricIndex,[2,4,5,7]) & T.corrReg==i);
%             figure
%             subplot(131)
%             plt.line(t.varReg,t.true_accuOrder,'split',t.metricIndex); ylabel('True order');
%             subplot(132)
%             plt.line(t.varReg,t.corrNoiseless,'split',t.metricIndex); ylabel('Noiseless corr');
%             subplot(133)
%             plt.line(t.varReg,t.true_misplaced,'split',t.metricIndex); ylabel('Positions missed');
%             
%             figure
%             subplot(221)
%             plt.line(t.varReg,t.true_accuOrder,'split',t.metricIndex); ylabel('True order');
%             subplot(222)
%             plt.line(t.varReg,t.spatial_accuOrder,'split',t.metricIndex); ylabel('Spatial order');
%             subplot(223)
%             plt.line(t.varReg,t.true_misplaced,'split',t.metricIndex); ylabel('Positions missed from truth');
%             subplot(224)
%             plt.line(t.varReg,t.spatial_misplaced,'split',t.metricIndex); ylabel('Positions missed from spatial struct');
%             
%             figure
%             subplot(121)
%             plt.line(t.varReg,t.tauTrue_NN,'split',t.metricIndex); ylabel('True order - neighbourhood');
%             subplot(122)
%             plt.line(t.varReg,t.tauSpatial_NN,'split',t.metricIndex); ylabel('Spatial order - neighbourhood');
%             
%             
            figure(500)
            subplot(numel(corrR),2,(find((corrR==i))-1)*2+1)
            plt.line(t.varReg,t.tauTrue_NN,'split',t.metricIndex); ylabel('True order');
            hold on; drawline(0,'dir','horz');
            title(sprintf('corr %1.1f',i));
            subplot(numel(corrR),2,(find((corrR==i))-1)*2+2)
            plt.line(t.varReg,t.tauSpatial_NN,'split',t.metricIndex); ylabel('Spatial order');
            hold on; drawline(0,'dir','horz');
        end
        
    case 'noise:within_plot' 
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        T = load(fullfile(baseDir,DNNname,'simulations_noise_within_newMetrics')); 
        %style.use('FourShade_cool');
        if strcmp(DNNname,'alexnet')
           t = getrow(T,T.dataType>1 & T.dataType~=5);
        else
            t = getrow(T,T.dataType>1 & T.dataType~=5);
        end
        style.use('FiveShade_cool');
        figure
        subplot(311)
        plt.line(t.varReg,t.corrNoiseless,'split',[t.dataType t.metricType]);
        xlabel('Noise'); ylabel('r(noiseless)'); title(DNNname);
        subplot(312)
        plt.line(t.varReg,abs(t.tauTrue_NN),'split',[t.dataType t.metricType]);
        xlabel('Noise'); ylabel('r(true structure)');
        subplot(313)
        plt.line(t.varReg,abs(t.tauTrue_all),'split',[t.dataType t.metricType]);
        xlabel('Noise'); ylabel('Comparison to true structure (corr)');
        
        figure
        subplot(121)
        plt.line(t.varReg,t.corrNoiseless,'split',[t.dataType t.metricType]);
        xlabel('Noise'); ylabel('r(noiseless)'); title(DNNname);
        subplot(122)
        plt.scatter(t.varReg,t.RDMconsist)
        xlabel('Noise');ylabel('RDM consistency'); title('fMRI consistency');
        % add lines at 1.9, 2.5, 2.7 (based on fMRI dataset)
    case 'noise:shared_plot_old' 
        DNNname = 'alexnet';
        dataName = {'uni','RDM-squared','RDM-sqrt','cRDM-squared','cRDM-sqrt','MVPD'};
        metricName = {'corr','cos','MVPD'};
        vararginoptions(varargin,{'DNNname'});
        
        T=load(fullfile(baseDir,DNNname,'simulations_noise_shared'));
        style.use('gray');
        t = getrow(T,ismember(T.dataType,[2,4,6]));
        for dt=unique(t.dataType)'
            t1 = getrow(t,t.dataType==dt);
            figure
            metric = unique(t1.metricType);
            for j=1:numel(metric)
                subplot(numel(unique(t1.metricType)),2,(j-1)*2+1)
                plt.line(t1.varReg,t1.corrNoiseless,'split',t1.corrReg,'subset',t1.metricType==metric(j)); ylabel('p(true structure)'); xlabel('overall noise');
                title(sprintf('%s-%s',dataName{dt},metricName{j}));
                subplot(numel(unique(t1.metricType)),2,(j-1)*2+2)
                plt.line(t1.varReg,abs(t1.tauTrue_all),'split',t1.corrReg,'subset',t1.metricType==metric(j)); ylabel('p(true structure) normalised'); xlabel('overall noise');
            end
        end
        figure
        idx=1;
        for dt=unique(t.dataType)'
            t1 = getrow(t,t.dataType==dt);
            metric = unique(t1.metricType);
            for j=1:numel(metric)
                subplot(2,4,idx)
                plt.line([t1.varReg>0 t1.varReg],t1.corrNoiseless,'split',t1.corrReg,'subset',t1.metricType==metric(j)&t1.corrReg<0.5); ylabel('r(noiseless structure)'); xlabel('overall noise');
                title(sprintf('%s-%s',dataName{dt},metricName{j}));
                subplot(2,4,4+idx)
                plt.line([t1.varReg>0 t1.varReg],abs(t1.tauSpatial_NN),'split',t1.corrReg,'subset',t1.metricType==metric(j)&t1.corrReg<0.5); ylabel('r(spatial noise structure)'); xlabel('overall noise');
                title(sprintf('%s-%s',dataName{dt},metricName{j}));
                idx = idx+1;
                plt.match('y');
            end
        end      
        corrR = [0,0.1,0.2,0.4];
        figure
        style.use('FourShade_cool');
        for i=1:length(corrR)
            subplot(2,length(corrR),i)
            plt.line([t.varReg>0 t.varReg],t.corrNoiseless,'split',[t.dataType t.metricType],'subset',t.corrReg==corrR(i),...
                'leg',{'RDM-sq-corr','cRDM-sq-corr','cRDM-sq-cos','MVPD'});
            if i==1
                ylabel('r(noiseless structure)');
            else
                ylabel(''); 
            end
            title(sprintf('correlated noise: %1.2f',corrR(i)));
            subplot(2,length(corrR),length(corrR)+i)
            plt.line([t.varReg>0 t.varReg],abs(t.tauSpatial_NN),'split',[t.dataType t.metricType],'subset',t.corrReg==corrR(i),...
                'leg',{'RDM-sq-corr','cRDM-sq-corr','cRDM-sq-cos','MVPD'});
            if i==1
                ylabel('r(spatial noise structure)'); 
            else
                ylabel(''); 
            end
            xlabel('overall noise');
            plt.match('y');
        end
    case 'noise:shared_confidence'
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        
        T=load(fullfile(baseDir,DNNname,'simulations_noise_shared_doubleCross_OHBM_NEW'));
        keyboard;
        style.use('gray');
        for dt=unique(T.dataType)'
            t1 = getrow(T,T.dataType==dt);
            figure
            metric = unique(t1.metricType);
            for j=1:numel(metric)
                subplot(numel(unique(t1.metricType)),2,(j-1)*2+1)
                plt.line(t1.varReg,t1.corrNoiseless,'split',t1.corrReg,'subset',t1.metricType==metric(j)); ylabel('p(true structure)'); xlabel('overall noise');
                subplot(numel(unique(t1.metricType)),2,(j-1)*2+2)
                plt.line(t1.varReg,abs(t1.tauSpatial_NN),'split',t1.corrReg,'subset',t1.metricType==metric(j)); ylabel('p(true structure) normalised'); xlabel('overall noise');
            end
        end
        keyboard;
        
        figure
        style.use('FourColor');
        subplot(221)
        plt.line(T.varReg,T.corrNoiseless,'split',[T.dataType,T.corrReg],'subset',T.metricType==1&T.dataType>2);
        title('Correlation - single vs. double crossval'); ylabel('correlation to noiseless');
        subplot(222)
        plt.line(T.varReg,T.corrNoiseless,'split',[T.dataType,T.corrReg],'subset',T.metricType==2&T.dataType>1);
        title('Cosine - single vs. double crossval'); ylabel('correlation to noiseless');
        subplot(223)
        plt.line(T.varReg,T.tauSpatial_NN,'split',[T.dataType,T.corrReg],'subset',T.metricType==1&T.dataType>2);
        ylabel('bias towards spatial noise'); xlabel('overall noise levels');
        subplot(224)
        plt.line(T.varReg,T.tauSpatial_NN,'split',[T.dataType,T.corrReg],'subset',T.metricType==2&T.dataType>1);
        ylabel('bias towards spatial noise'); xlabel('overall noise levels');
        
    case 'noise:plot_perReg'
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        
        T=load(fullfile(baseDir,DNNname,'simulations_noise_perReg_OHBM2'));
        nLayer = size(T.RDMconsist_all,2);
        mColor = cell(1,nLayer);
        mCol = hsv(nLayer);
        for i=1:nLayer
            mColor{i}=mCol(i,:);
        end 
        figure
        for d=1:2
            subplot(3,2,d)
            plt.bar(T.dataType,T.corrNoiseless,'split',T.metricType,'subset',T.dataType>1&T.noiseType==d);
            subplot(3,2,d+2)
            plt.bar(T.dataType,T.tauTrue_NN,'split',T.metricType,'subset',T.dataType>1&T.noiseType==d);
            subplot(3,2,d+4)
            plt.bar(T.dataType,T.tauTrue_all,'split',T.metricType,'subset',T.dataType>1&T.noiseType==d);
        end
        % try also isomap
        keyboard;
        for d=1:2
            figure
            for idx=1:4
                if idx==1
                    D = rsa_squareRDM(mean(T.RDM(T.dataType==2&T.metricType==1&T.noiseType==d,:),1));
                elseif idx==2
                    D = rsa_squareRDM(mean(T.RDM(T.dataType==3&T.metricType==1&T.noiseType==d,:),1));
                elseif idx==3
                    D = rsa_squareRDM(mean(T.RDM(T.dataType==3&T.metricType==2&T.noiseType==d,:),1));
                else
                    D = rsa_squareRDM(mean(T.RDM(T.dataType==4&T.noiseType==d,:),1));
                end
                %D = D(2:end,2:end);
                if idx<3
                    [mX,mp] = topology_estimate(D,2,2); % submit to topology function
                else
                    [mX,mp] = topology_estimate(D,2,2); % submit to topology function
                end
                subplot(4,2,(idx-1)*2+1)
                imagesc(D); colormap hot;
                W = full(mp.D);
                [r,c,val] = find(W);
                val = val./max(val); % renormalize
                subplot(4,2,(idx-1)*2+2)
                hold on;
                for i=1:length(r)
                    plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',1,'Color',repmat(val(i),3,1)./(max(val)+0.1));
                end
                scatterplot(mX(:,1),mX(:,2),'label',(1:nLayer),'split',(1:nLayer)','markercolor',mColor,'markertype','.','markersize',30);
                axis equal; axis off;
            end
        end
    case 'noise:simulate_perReg_old'
        % simulate noise per region (alternating) - low high low high...
        nPart       = 8;
        nSim        = 100;
        nUnits      = 500;
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nPart','nSim','dataType','DNNname'});
        % initialize
        dataInd     = [1 1 2 2 3 3 4 4 5 5 6 6 6 7 7 7 8]; % anzelotti as 8
        alphaInd    = [1 2 1 2 1 2 1 2 1 2 3 4 5 3 4 5 6];
        RDMInd      = [1 2 3 4 5 6 3 4 5 6 7 8 9 7 8 9 0];
        dirMetrics  = {'scaleDist','diagDist','eigComp'};
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        nCond = size(act{1},1);
        nLayer = size(act,1);
        % first normalize activities
        act = DNN_connect('HOUSEKEEPING:normalizeUnits',act);
        trueOrder = squareform(pdist((1:nLayer)'));
        
        varReg = repmat([1.7 3.4],1,nLayer/2); % modulate
        corrReg = 0;
        VV=[];
        for r=corrReg       % correlated noise
            for n=1:nSim    % number of simulations
                % here subsample + repeat the true pattern for each partition
                act_subsets = cell(nLayer,1);
                data = cell(nLayer,1); Data=data;
                for i=1:nLayer
                    rUnits = randperm(size(act{i},2)); % randomise the order of units
                    act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                    data{i} = repmat(act_subsets{i},nPart,1);
                    Data(i) = addSharedNoise(data(i),varReg(i),corrReg,'within');
                end
                tElapsed=tic;
                [RDMconsist,RDMconsist_all] = rdmConsist(Data,nPart,size(act_subsets{1},1));
                [fD{1},fD{2},fD{3},fD{4},fD{5},fD{6},fD{7}]=DNN_connect('firstLevel:calculate',Data,nCond);
                % order: uni, RDM, RDM_sqrt, cRDM, cRDM_sqrt, G, cG
                A{1} = DNN_connect('secondLevel:transformG',fD{6});
                A{2} = DNN_connect('secondLevel:transformG',fD{7});
                % cycle around all combinations of data / metrics
                for i=1:length(dataInd)
                    if dataInd(i) < 6
                        t = DNN_connect('secondLevel:calcDist',fD{dataInd(i)},aType{alphaInd(i)});
                        T = rsa_squareRDM(t.dist');
                        V.corrNoiseless = corr(TD.dist(RDMInd(i),:)',rsa_vectorizeRDM(T)');
                    elseif dataInd(i) > 5 && dataInd(i) < 8
                        T = rsa_squareIPMfull(A{dataInd(i)-5}.(dirMetrics{alphaInd(i)-2})');
                        T = T+T'; % symmetrize
                        V.corrNoiseless = corr(TD.dist(RDMInd(i),:)',rsa_vectorizeRDM(T)');
                    else % Anzellotti
                        T = anzellottiDist(Data,nPart,size(act{1},1));
                        V.corrNoiseless = NaN;
                    end
                    NN              = construct_neighbourhood(T);
                    V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                    V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(T)');  % from original distances
                    V.RDM           = rsa_vectorizeRDM(T);
                    V.RDMconsist    = RDMconsist;
                    V.RDMconsist_all= RDMconsist_all;
                    V.dataType      = dataInd(i); % uni / multi / directional
                    V.metricType 	= alphaInd(i); % corr / cos / eigStd ...
                    V.metricIndex   = i;
                    V.numSim        = n;
                    V.corrReg       = r;
                    V.varReg        = varReg;
                    VV=addstruct(VV,V);
                end
                fprintf('%d.',n);
                toc(tElapsed);
            end
        end
        fprintf('\nFinished correlation %d.\n\n',r);
        save(fullfile(baseDir,DNNname,'simulations_noise_perReg'),'-struct','VV');
        fprintf('\nDone simulations %s: perReg\n',DNNname);
    case 'noise:plot_old'
        noiseType = 'within_low';
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname','noiseType'});
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)));
        % for comparison with fMRI noise
        R = load(fullfile(baseDir,'RDMreplicability_correlation'));
        rep = nanmean(R.RDMreplicability_subj_roi,1);
        % for comparing consistency
        N = tapply(T,{'varReg'},{'RDMconsist'});
        p1=N.varReg(N.RDMconsist<rep(1));
        p2=N.varReg(N.RDMconsist<mean(rep([2,3])));
        p3=N.varReg(N.RDMconsist<mean(rep([4,5])));
        
        for i=unique(T.dataType)'
            figure
            if i~=8
                subplot(311)
                plt.line(T.varReg,T.corrNoiseless,'split',T.metricType,'subset',T.dataType==i);
                hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
                xlabel('Noise'); ylabel('r(noiseless)');
            end
            subplot(312)
            plt.line(T.varReg,T.tauTrue_NN,'split',T.metricType,'subset',T.dataType==i);
            hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
            xlabel('Noise'); ylabel('r(true structure)');
            
            subplot(313)
            plt.line(T.varReg,T.tauTrue_all,'split',T.metricType,'subset',T.dataType==i);
            hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
            xlabel('Noise'); ylabel('Comparison to true structure (corr)');
        end
    case 'noise:plot2'
        % more specific contrasts
        noiseType = 'within_low';
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname','noiseType'});
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)));
        % for comparison with fMRI noise
       % R = load(fullfile(baseDir,'RDMreplicability_correlation'));
     %   rep = nanmean(R.RDMreplicability_subj_roi,1);
        % for comparing consistency
     %   N = tapply(T,{'varReg'},{'RDMconsist'});
    %    p1=N.varReg(N.RDMconsist<rep(1));
    %    p2=N.varReg(N.RDMconsist<mean(rep([2,3])));
    %    p3=N.varReg(N.RDMconsist<mean(rep([4,5])));
        keyboard;
        style.use('AlterShade');
        figure % here cosine / corr for multivariate - ncv
        subplot(311)
        plt.line([T.varReg>1.5 T.varReg],T.corrNoiseless,'split',[T.dataType T.metricType],'subset',ismember(T.dataType,[2,3]));
       % hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
        xlabel('Noise'); ylabel('r(noiseless)');
        subplot(312)
        plt.line([T.varReg>1.5 T.varReg],T.tauTrue_NN,'split',[T.dataType T.metricType],'subset',ismember(T.dataType,[2,3]));
       % hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
       xlabel('Noise'); ylabel('true neighbourhood');
        subplot(313)
        plt.line([T.varReg>1.5 T.varReg],T.tauTrue_all,'split',[T.dataType T.metricType],'subset',ismember(T.dataType,[2,3]));
        xlabel('Noise'); ylabel('true structure');
        keyboard;
        style.use('AlterShade3');
        figure % here cosine / corr for multivariate - ncv
        subplot(311)
        plt.line([T.varReg>1.5 T.varReg],T.corrNoiseless,'split',[T.dataType T.metricType],'subset',ismember(T.dataType,[4,5,8]));
      %  hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
        xlabel('Noise'); ylabel('r(noiseless)');
        subplot(312)
        plt.line([T.varReg>1.5 T.varReg],T.tauTrue_NN,'split',[T.dataType T.metricType],'subset',ismember(T.dataType,[4,5,8]));
       % hold on; drawline([p1(1),p2(1),p3(1)],'dir','vert','color',[.7 .7 .7]);
        xlabel('Noise'); ylabel('true neighbourhood');
        subplot(313)
        plt.line([T.varReg>1.5 T.varReg],T.tauTrue_all,'split',[T.dataType T.metricType],'subset',ismember(T.dataType,[4,5,8]));
        xlabel('Noise'); ylabel('true structure');
    case 'noise:plot_perReg_old'
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'DNNname'});
        T = load(fullfile(baseDir,DNNname,'simulations_noise_perReg'));
        figure
        subplot(311)
        plt.bar(T.dataType,T.corrNoiseless,'split',T.metricType); ylabel('noiseless');
        subplot(312)
        plt.bar(T.dataType,T.tauTrue_NN,'split',T.metricType); ylabel('truth-neighbourhood');
        subplot(313)
        plt.bar(T.dataType,T.tauTrue_all,'split',T.metricType); ylabel('truth-all');
    case 'noise:shared_normalise'
        noiseType = 'shared_helpful';
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname','noiseType'});
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)));
        DD=[];
        for i=unique(T.dataType)'
            for j=unique(T.metricType)'
                T1 = getrow(T,T.dataType==i & T.metricType==j);
                % overall normalisation to varReg / corrReg = 0
                t = mean(T1.tauTrue_all(T1.varReg==0 & T1.corrReg==0));
                T1.tauTrue_all_norm1 = T1.tauTrue_all./t;
                norm2=[];
                for n=unique(T.varReg)' % per noise level (varReg)
                    T2 = getrow(T1,T1.varReg==n);
                    t2 = mean(T2.tauTrue_all(T2.corrReg==0));
                    norm2 = [norm2;T2.tauTrue_all./t2];
                end
                T1.tauTrue_all_norm2 = norm2;
                DD = addstruct(DD,T1);
            end
        end
        save(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)),'-struct','DD');
    case 'noise:plot_shared_old'
        noiseType = 'shared_helpful';
        DNNname = 'alexnet';
        vararginoptions(varargin,{'DNNname','noiseType'});
        tt = {'uni-corr','uni-cos','RDM-corr-squared','RDM-cos-squared','RDM-corr-sqrt','RDM-cos-sqrt',...
            'cRDM-corr-squared','cRDM-cos-squared','cRDM-corr-sqrt','cRDM-cos-sqrt',...
            'multiDepend'};
        dt = [1,2,3,4,5,8];
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)));
        
        style.use('gray');
        idx=1;
        for i=dt
            T1 = getrow(T,T.dataType==i);
            for j=unique(T1.metricType)'
                t1 = getrow(T1,T1.metricType==j);
                [f1,~,~]=pivottable(t1.corrReg,[t1.varReg],t1.tauTrue_all,'mean');
                [f2,~,~]=pivottable(t1.corrReg,[t1.varReg],t1.tauTrue_all_norm1,'mean'); % two ways of normalising
                [f3,~,~]=pivottable(t1.corrReg,[t1.varReg],t1.tauTrue_all_norm2,'mean');
                figure
                subplot(321)
                imagesc(flipud(f1)); title(sprintf('r(true structure) - %s',tt{idx}));
                ylabel('correlated noise');
                subplot(323)
                imagesc(flipud(f2)); title(sprintf('normalised overall - %s',tt{idx}));
                ylabel('correlated noise');
                subplot(325)
                imagesc(flipud(f3)); title(sprintf('normalised per noise level - %s',tt{idx}));
                ylabel('correlated noise'); xlabel('overall noise');
                colormap hot;
                t1 = getrow(t1,round(t1.corrReg,2)~=0.6);
                subplot(322)
                plt.line(t1.varReg,t1.tauTrue_all,'split',t1.corrReg); ylabel('p(true structure)'); xlabel('overall noise');
                subplot(324)
                plt.line(t1.varReg,t1.tauTrue_all_norm1,'split',t1.corrReg); ylabel('p(true structure) normalised'); xlabel('overall noise');
                hold on; drawline(0.5,'dir','horz');
                subplot(326)
                plt.line(t1.varReg,t1.tauTrue_all_norm2,'split',t1.corrReg); ylabel('p(true structure) normalised'); xlabel('overall noise');
                idx=idx+1;
            end
        end
    case 'noise:plot_shared_var'
        DNNname     = 'alexnet';
        noiseType = 'shared_harmful';
        vararginoptions(varargin,{'DNNname','noiseType'});
        dt = [1,2,3,4,5,8];
        T = load(fullfile(baseDir,DNNname,sprintf('simulations_noise_%s',noiseType)));
        
        T = getrow(T,ismember(T.dataType,dt));
        DD = [];
        for i=unique(T.dataType)'
            T1 = getrow(T,T.dataType==i);
            for j=unique(T1.metricType)'
                % overall normalisation to varReg / corrReg = 0
                for n=unique(T1.varReg)' % per noise level (varReg)
                    T2 = getrow(T1,T1.dataType==i & T1.metricType==j & T1.varReg==n);
                    D.var       = var(T2.tauTrue_all);
                    D.var_norm1 = var(T2.tauTrue_all_norm1);
                    D.var_norm2 = var(T2.tauTrue_all_norm2);
                    D.varReg = n;
                    D.metricType = j;
                    D.dataType = i;
                    DD = addstruct(DD,D);
                end
            end
        end
        figure
        style.use('Alter3');
        subplot(311)
        plt.line(DD.varReg,DD.var,'split',[DD.dataType DD.metricType],'subset',ismember(DD.dataType,[4,5,8]));
        xlabel('Overall noise'); ylabel('Variance of estimate');
        subplot(312)
        plt.line(DD.varReg,DD.var_norm1,'split',[DD.dataType DD.metricType],'subset',ismember(DD.dataType,[4,5,8]));
        xlabel('Overall noise'); ylabel('Variance of norm1 estimate');
        subplot(313)
        plt.line(DD.varReg,DD.var_norm2,'split',[DD.dataType DD.metricType],'subset',ismember(DD.dataType,[4,5,8]));
        xlabel('Overall noise'); ylabel('Variance of norm2 estimate');
    case 'noise:plot_shared_confidence'
        DNNname = 'alexnet';
        T = load(fullfile(baseDir,DNNname,'simulations_noise_shared_doubleCross_OHBM'));
        C = load(fullfile(baseDir,DNNname,'simulations_noise_confidence_doubleCross_OHBM'));
        style.use('gray');
        for i=unique(T.dataType)'
            t=getrow(T,T.dataType==i & T.corrReg<0.8);
            figure
            metric = unique(t.metricType)';
            for j=1:numel(metric)
                subplot(numel(metric),3,(j-1)*3+1)
                plt.line(t.varReg,t.corrNoiseless,'split',t.corrReg,'subset',t.metricType==metric(j));
                hold on; drawline(1,'dir','horz');
                title('true structure'); ylabel('Pearson r');
                subplot(numel(metric),3,(j-1)*3+2)
                plt.line(t.varReg,t.tauSpatial_NN,'split',t.corrReg,'subset',t.metricType==metric(j));
                hold on; drawline(0,'dir','horz');
                title('spatial noise structure'); ylabel('Pearson r');
                subplot(numel(metric),3,(j-1)*3+3)
                plt.line(C.varReg,mean(C.conf,2),'split',C.corrReg,'subset',C.dataType==i & C.metricType==metric(j) & C.corrReg<0.8);
                hold on; drawline(0,'dir','horz');
                title('uncertainty'); ylabel('variance');
            end
        end
        t = getrow(T,T.dataType==4);
        figure
        style.use('TwoShade');
        plt.scatter(t.RDMconsist,mean(C.conf,2),'split',t.metricType,'leg',{'corr','cos'});
        xlabel('RDM consistency (per region)'); ylabel('variance in connectivity estimate');
    
    case 'noiseless:subsetCond'
        % validate topology in the noiseless case by subsampling conditions
        % use the normalized activation units here
       % validate topology in the noiseless case by subsampling units
        nSim        = 100; % number of simulations
        nCond       = 50; % number of units to sample at a time
        DNNname     = 'alexnet';
        dirMetrics  = {'scaleDist','diagDist','eigComp'}; % directional metrics to consider
        vararginoptions(varargin,{'nSim','nUnits','nDim','nNeigh','DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        numLayer = size(act,1);
        order = 1:numLayer; trueOrder = squareform(pdist(order'));
        
        dataType    = [1 1 2 2 3 3 repmat(4,1,length(dirMetrics))]; % uni / RDM / RDM_sqrt / directional
        metricType  = [1 2 1 2 1 2 3:2+length(dirMetrics)]; % corr / cos / scaleDist...
        VV = [];
        for n=1:nSim % simulations
            tStart=tic;
            rCond = randperm(size(act{1},1)); % randomise the conditions
            act_subsets = cell(numLayer,1);
            for i=1:numLayer
                act_subsets{i} = act{i}(rCond(1:nCond),:);
            end
            act_subsets = DNN_connect('HOUSEKEEPING:normalizeUnits',act_subsets);
            %% 1) here estimate first level
            [fD{1},fD{2},fD{3},~,~,fD{4},~]=DNN_connect('firstLevel:calculate',act_subsets,size(act_subsets{1},1)); % order: uni, RDM, cRDM, G, cG            
            %% 2) estimate second level metrics (between RDMs, Gs)
            % calculate transformation matrices T between Gs
            A = DNN_connect('secondLevel:transformG',fD{4});
            %% 3) estimate topology - neighbourhood, then relate to true order
            for m = 1:length(metricType)
                d = dataType(m);
                if d<4 % uni / multivariate - corr / cos
                    t = DNN_connect('secondLevel:calcDist',fD{dataType(m)},aType{metricType(m)});
                    D = rsa_squareRDM(t.dist');
                else % directional
                    t = rsa_squareIPMfull(A.(dirMetrics{m-6})');
                    D = t+t'; % make it symmetric
                end 
                V.corrNoiseless = corr(TD.dist(m,:)',rsa_vectorizeRDM(D)');
                NN              = construct_neighbourhood(D);
                V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(D)');  % from original distances
                V.metric        = m;
                V.dataType      = d; % uni / multi / directional
                V.metricType    = metricType(m); % corr / cos / eigStd ...
                V.numSim        = n;
                VV=addstruct(VV,V);
            end
            fprintf('%d. simulation done...',n);
            toc(tStart);
        end
        % save here
        save(fullfile(baseDir,DNNname,'validate_noiseless_subsetCond_v2'),'-struct','VV'); 
        fprintf('Done subsampling conditions: %s\n\n',DNNname);
    case 'noiseless:subsetUnit'
       % validate topology in the noiseless case by subsampling units
        nSim        = 100; % number of simulations
        nUnits      = 500; % number of units to sample at a time
        DNNname     = 'alexnet';
        dirMetrics  = {'scaleDist','diagDist','eigComp'}; % directional metrics to consider
        vararginoptions(varargin,{'nSim','nUnits','nDim','nNeigh','DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        TD = load(fullfile(baseDir,DNNname,sprintf('%s_trueDist',DNNname))); % true distances
        numLayer = size(act,1);
        order = 1:numLayer; trueOrder = squareform(pdist(order'));
        
        dataType    = [1 1 2 2 3 3 repmat(4,1,length(dirMetrics))]; % uni / RDM / RDM_sqrt / directional
        metricType  = [1 2 1 2 1 2 3:2+length(dirMetrics)]; % corr / cos / scaleDist...
        VV = [];
        for n=1:nSim % simulations
            tStart=tic;
            act_subsets = cell(numLayer,1);
            for i=1:numLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
            end
            act_subsets = DNN_connect('HOUSEKEEPING:normalizeUnits',act_subsets);
            %% 1) here estimate first level
            [fD{1},fD{2},fD{3},~,~,fD{4},~]=DNN_connect('firstLevel:calculate',act_subsets,size(act_subsets{1},1)); % order: uni, RDM, cRDM, G, cG            
            %% 2) estimate second level metrics (between RDMs, Gs)
            % calculate transformation matrices T between Gs
            A = DNN_connect('secondLevel:transformG',fD{4});
            %% 3) estimate topology - neighbourhood, then relate to true order
            for m = 1:length(metricType)
                d = dataType(m);
                if d<4 % uni / multivariate - corr / cos
                    t = DNN_connect('secondLevel:calcDist',fD{dataType(m)},aType{metricType(m)});
                    D = rsa_squareRDM(t.dist');
                else % directional
                    t = rsa_squareIPMfull(A.(dirMetrics{m-6})');
                    D = t+t'; % make it symmetric
                end 
                V.corrNoiseless = corr(TD.dist(m,:)',rsa_vectorizeRDM(D)');
                NN              = construct_neighbourhood(D);
                V.tauTrue_NN    = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(NN)'); % from neighbourhood
                V.tauTrue_all   = corr(rsa_vectorizeRDM(trueOrder)',rsa_vectorizeRDM(D)');  % from original distances
                V.metric        = m;
                V.dataType      = d; % uni / multi / directional
                V.metricType    = metricType(m); % corr / cos / eigStd ...
                V.numSim        = n;
                VV=addstruct(VV,V);
            end
            fprintf('%d. simulation done...',n);
            toc(tStart);
        end
        % save here
        save(fullfile(baseDir,DNNname,'validate_noiseless_subsetUnit_v2'),'-struct','VV'); 
        fprintf('Done subsampling units: %s\n\n',DNNname);
    case 'noiseless:subsetUnit_anzellotti'
        nSim        = 20; % number of simulations
        nUnits      = 500; % number of units to sample at a time
        nPart       = 8;
        n_dim       = 2;
        n_neigh     = 1;
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'nSim','nUnits','nDim','nNeigh','DNNname'});
        load(fullfile(baseDir,DNNname,sprintf('%s_activations',DNNname)));
        numLayer = size(act,1);

        VV = [];
        fprintf('Simulations:\n');
        for n=1:nSim % simulations
            act_subsets = cell(numLayer,1);
            for i=1:numLayer
                rUnits = randperm(size(act{i},2)); % randomise the order of units
                act_subsets{i} = act{i}(:,rUnits(1:nUnits));
                data{i,:} = repmat(act_subsets{i},nPart,1);
            end
            D = anzellottiDist(data,nPart,size(act{1},1));       % Anzellotti
            V.RDM = rsa_vectorizeRDM(D);
            V.numSim        = n;
            VV=addstruct(VV,V);
            fprintf('%d.',n);
        end
        D = rsa_squareRDM(nanmean(VV.RDM,1));
        % topology
        mColor = cell(1,numLayer);
        mCol = hsv(numLayer);
        for i=1:numLayer
            mColor{i}=mCol(i,:);
        end
        [mX,mp] = topology_estimate(D,n_dim,n_neigh); % submit to topology function
        figure
        subplot(121)
        betCol = [245 132 132]./255;
        anchorCols = [1 1 1; betCol; 1 0 0];
        cols = colorScale(anchorCols,256,1);
        imagesc(D); colormap(gca,cols);
        title('Anzellotti RDM');
        subplot(122)
        hold on;
        W = full(mp.D);
        [r,c,val] = find(W);
        val = val./max(val); % renormalize
        for i=1:length(r)
            plot([mX(r(i),1),mX(c(i),1)],[mX(r(i),2),mX(c(i),2)],'LineWidth',1,'Color',repmat(val(i),3,1)./(max(val)+0.1));
        end
        scatterplot(mX(:,1),mX(:,2),'label',(1:numLayer),'split',(1:numLayer)','markercolor',mColor,'markertype','.','markersize',30);
        axis equal; axis off;
        title('Anzellotti isomap');
        keyboard;
        saveas(gcf,fullfile(baseDir,DNNname,'anzellotti_reconstruction.pdf'));
    case 'noiseless:plot_subset'
        subsetType = 'Cond'; % Cond or Unit
        DNNname     = 'alexnet';
        vararginoptions(varargin,{'subsetType','DNNname'});
        T = load(fullfile(baseDir,DNNname,sprintf('validate_noiseless_subset%s_v2',subsetType)));
        tickLab = {'uni-corr','uni-cos','multi-corr','multi-cos',...
            'multi-corr-sqrt','multi-cos-sqrt','G-scaleDist','G-diagDist','G-eigComp'};
        warning off
        
        figure
%         subplot(311)
%         plt.bar(T.dataType,T.tauTrue_all,'split',T.metricType);
%         hold on; drawline(1,'dir','horz');
%         set(gca,'XTickLabel',tickLab); ylabel('Comparison to true ordering');
 
        subplot(211)
        plt.bar(T.dataType,T.tauTrue_NN,'split',T.metricType);
        hold on; drawline(1,'dir','horz');
        set(gca,'XTickLabel',tickLab); ylabel('True ordering');
        title(sprintf('subsampling %s - %s',subsetType, DNNname)); legend off;
        subplot(212)
        plt.bar(T.dataType,T.corrNoiseless,'split',T.metricType);
        hold on; drawline(1,'dir','horz');
        set(gca,'XTickLabel',tickLab); ylabel('Complete noiseless');
        legend off;
    
    case 'HOUSEKEEPING:normalizeUnits'
        % here normalize the activation in units (so the variance of signal
        % is comparable across layers)
        act = varargin{1};
        numLayer = size(act,1);
        actN = cell(numLayer,1);
        for i=1:numLayer
            % old version
        %    actN{i}=bsxfun(@minus,act{i},mean(act{i},1));  % first here remove the mean activation
        %    actN{i}=actN{i}./max(max(actN{i}));
            % new version
            actN{i}=bsxfun(@minus,act{i},mean(act{i},2));
            actN{i}=bsxfun(@rdivide,act{i},std(act{i},[],2));
            % alternative version
        %    actN{i}=bsxfun(@minus,act{i},mean(act{i},1));
        %    actN{i}=bsxfun(@rdivide,act{i},std(act{i},[],1));
        end
        varargout{1}=actN;      
    case 'HOUSEKEEPING:shuffled_structure'
        % calculate the correct structure for shuffled order
        act=varargin{1};
        numLayer=size(act,1);
        randOrder = randperm(numLayer);
        C    = zeros(numLayer);
        actS = cell(size(act)); 
        for i=1:length(randOrder)
            actS{i} = act{randOrder(i)};
            C(i,:) = abs(randOrder-randOrder(i));
        end
        varargout{1}=actS;
        varargout{2}=C;
        varargout{3}=randOrder;
    case 'HOUSEKEEPING:removeRunMean'
        act = varargin{1};
        nPart = varargin{2};
        nCond = varargin{3};
        partVec = kron((1:nPart)',ones(nCond,1));
        numLayer = size(act,1);
        actN = cell(numLayer,1);
        for i=1:numLayer
            actN{i} = [];
            for j=1:max(partVec)
                actN{i} = [actN{i}; bsxfun(@minus,act{i}(partVec==j,:),mean(act{i}(partVec==j,:),1))];
            end
        end
        varargout{1}=actN;    


    case 'run_job'
        rep_connect('noise:simulate','nSim',500);
        rep_connect('noise:simulate','noiseType','within_oneNoisy','nSim',500); 
        rep_connect('noise:simulate','noiseType','shared_harmful','nSim',100,'corrReg',0:.1:.9,'varReg',0:.25:4);
        
    otherwise 
        fprintf('no such case!\n');

end
end
%% Local functions
function dircheck(dir)
% Checks existance of specified directory. Makes it if it does not exist.

if ~exist(dir,'dir');
    %warning('%s didn''t exist, so this directory was created.\n',dir);
    mkdir(dir);
end
end
function [eigT,eigRange,eigStd,eigComp,eigComp2] = eig_complexity(M)
%function eigC = eig_complexity(M)
%calculates eigenvalues of matrix M
%and derivatives
eigT        = eig(M)';
eigT        = round(eigT,10); % to ensure that negative values are really neg, not 0
if any(eigT<0)
    eigT=eigT(eigT>0);
end
eigRange    = max(eigT)-min(eigT);
eigStd      = std(eigT);
eigComp     = 1-((sum(eigT))^2/(sum(eigT.^2)))/size(M,1);
eigComp2    = ((sum(eigT))^2/(sum(eigT.^2)));
end
function [data,spatOrder] = addSharedNoise(data,Var,r,noiseType)
% input: 
% data - datasets
% alpha - variance
% r - correlation
% noiseType - allEqual or neighbours
    nDataset = size(data,1);
    nTrials = size(data{1},1);
    nVox    = size(data{1},2);
    spatOrder = [];
    % add shared noise (shared within / across regions)
    Z = normrnd(0,1,nTrials,nDataset*nVox);
   % Z = Z./max(max(Z)); % removed Sept 17 2019
    switch noiseType
        case 'within'
            Zn = Z;
        case 'within_oneNoisy'
            % within region noise with one noisier region
            reg = randi(nDataset);
           % reg = 3;
            Zn = Z;
        case 'allEqual'
            A = eye(nDataset)*Var+(ones(nDataset)-eye(nDataset))*Var*r;
            P = kron(A,eye(nVox));
            Zn = Z*sqrtm(P);     % shared noise matrix across reg
        case 'neighbours'
            % first structured shared noise
            kernelWidth = 500; % decide on the kernel width
            t           = 1:nVox*nDataset;
            dist        = pdist(t');
            noiseKernel = exp((-0.5*dist)/kernelWidth);
            P           = squareform(noiseKernel);
            %   Zn          = Z*real(sqrtm(P));
            Zn          = Z*P;
            % modulate the relative strength of the shared noise 
            Zn = Zn./max(max(Zn));
            Zn = Z.*(1-r)+(Zn.*r);
        case 'shared_helpful'
            % first structured shared noise
            kernelWidth = 500; % decide on the kernel width
            t           = 1:nVox*nDataset;
            dist        = pdist(t');
            noiseKernel = exp((-0.5*dist)/kernelWidth);
            P           = squareform(noiseKernel);
            %   Zn          = Z*real(sqrtm(P));
            Zn          = Z*P;
            % modulate the relative strength of the shared noise 
            Zn = Zn./max(max(Zn));
            Zn = Z.*(1-r)+(Zn.*r);
        case 'shared_harmful'
            % first structured shared noise
            kernelWidth = 500; % decide on the kernel width
            t           = 1:nVox*nDataset;
            dist        = pdist(t');
            noiseKernel = exp((-0.5*dist)/kernelWidth);
            P           = squareform(noiseKernel);
            Zn          = Z*P;
            % modulate the relative strength of the shared noise 
            Zn = Zn./max(max(Zn));
            Zn = Z.*(1-r)+(Zn.*r);
            spatOrder = randperm(8);
            Zn2 = zeros(size(Zn));
            for j=1:8
                Zn2(:,(j-1)*nVox+1:j*nVox) = Zn(:,(spatOrder(j)-1)*nVox+1:spatOrder(j)*nVox);
            end
            Zn = Zn2;
    end
  %  Zn = Zn./max(max(Zn)); % removed Sept 17 2019
    for i=1:nDataset
        if strcmp(noiseType,'within_oneNoisy') && i==reg
            data{i} = data{i} + 3*Var.*Zn(:,(i-1)*nVox+1:i*nVox); % make 3x as noisy
        else
            data{i} = data{i} + Var.*Zn(:,(i-1)*nVox+1:i*nVox);
        end
        % d = DNN_connect('HOUSEKEEPING:normalizeUnits',data(i)); % new - no
        % need to double normalize % removed Sept 17 2019
       % data{i} = d{1}; % new
        % data{i} = data{i}./max(max(data{i}));
    end
end
function [RDMconsist_av, RDMconsist_all, Conf_corr, Conf_cos] = rdmConsist(Data,nPart,nCond)
% function [RDMconsist_av RDMconsist_all] = rdmConsist(data,nPart,nCond);
% calculate the consistency of RDMs across runs 
% outputs: 1) across all layers, 2) per layer
numLayer = size(Data,1);
RDMconsist_all = zeros(1,numLayer);
partVec = kron((1:nPart)',ones(nCond,1));
% contrast matrix for G->distances
C = indicatorMatrix('allpairs',1:nCond);
X = indicatorMatrix('identity_p',[1:nCond]);
H = eye(nCond)-ones(nCond)./nCond;
for i=1:numLayer
    nVox = size(Data{i},2);
    % calculate mean activation
    t=Data{i};
    for j=1:nPart % first remove the mean of each run
        t(partVec==j,:)=bsxfun(@minus,Data{i}(partVec==j,:),mean(Data{i}(partVec==j,:),1));
        D           = pinv(X)*t(partVec==j,:);
        G           = D*D'/nVox;
        G           = H*G*H';
        RDM(j,i,:)  = diag(C*G*C')';
    end
    RDMconsist_all(i)  = mean(rsa_vectorizeRDM(corr(squeeze(RDM(:,i,:))')));
end
RDMconsist_av = mean(RDMconsist_all);
% confidence for correlation and cosine - run estimates
run_cos = zeros(numLayer*(numLayer-1)/2,max(partVec));
run_corr = zeros(numLayer*(numLayer-1)/2,max(partVec));
for p=1:max(partVec)
    rdm = squeeze(RDM(p,:,:));
    % additional step for correlation - first remove the mean
    rdmCR = normalizeX(bsxfun(@minus,rdm,mean(rdm,2)));
    rdmCS = normalizeX(rdm);
    tmpCR = rdmCR*rdmCR'; % correlation across RDMs
    tmpCS = rdmCS*rdmCS';
    run_corr(:,p)= 1-rsa_vectorizeRDM(tmpCR)'; % distances
    run_cos(:,p) = 1-rsa_vectorizeRDM(tmpCS)'; % distances
end
Conf_corr = (var(run_corr,[],2))';
Conf_cos  = (var(run_cos,[],2))';
end
function [R2_MVPD,r_MVPD] = anzellottiDist(data,nPart,nCond)
% function dist = anzellottiDist(data,partVec,condVec)
% calculates a relationship between data of different regions
% for now the distance output is 1-R2 
nData   = size(data,1);
partVec = kron((1:nPart)',ones(nCond,1));          
condVec = kron(ones(nPart,1),(1:nCond)');
ind     = indicatorMatrix('allpairs',1:nData);
R2_mvpd = zeros(size(ind,1),1);
r_mvpd  = zeros(size(ind,1),1);
for i=1:size(ind,1)
    ind1 = find(ind(i,:)==1);
    ind2 = find(ind(i,:)==-1);
    [R2_mvpd(i),r_mvpd(i)] = multiDependVox(data{ind2},data{ind1},partVec,condVec,'type','reduceAB');
end
R2_MVPD= rsa_squareRDM(R2_mvpd');
r_MVPD = rsa_squareRDM(r_mvpd');
end
function [lCKA,corrRDM]     = calcCKA(Data,nPart,nCond,mode)
% -------------------------------------------------------------------------
if ~ismember(mode,{'average','runwise','crossval'})% first check if mode valid
    error('Wrong mode, options: average, runwise, crossval');
end
nReg = size(Data,1);
partVec = kron((1:nPart)',ones(nCond,1));          
condVec = kron(ones(nPart,1),(1:nCond)');
ind = indicatorMatrix('allpairs',1:nReg);
H = eye(nCond)-ones(nCond)./nCond;
C = indicatorMatrix('allpairs',1:nCond);
lCKA = zeros(nReg); corrRDM = zeros(nReg); %corrRDM_pw = zeros(nReg);
X = indicatorMatrix('identity_p',condVec);
for l=1:nReg
    D{l} = pinv(X)*Data{l};
    G1 = D{l}*D{l}'/size(D{l},2);
    G{l} = H*G1*H'; % double centered
    RDM{l} = squareform(diag(C*G{l}*C')');
end
for r=1:size(ind,1)
    reg1 = find(ind(r,:)==1);
    reg2 = find(ind(r,:)==-1);
    % select which runs / per run or crossvalidated
    if strcmp(mode,'average')
        c = corr(rsa_vectorizeIPMfull(G{reg1}')',rsa_vectorizeIPMfull(G{reg2}')');
        dist = corr(rsa_vectorizeIPMfull(RDM{reg1})',rsa_vectorizeIPMfull(RDM{reg2})');
    else
        % per run
        c1 = zeros(1,nPart); dist1 = zeros(1,nPart);
        for i=1:nPart
            D{1} = Data{reg1}(partVec==i,:);
            if strcmp(mode,'runwise')
                D{2} = Data{reg2}(partVec==i,:);
            else
                data = Data{reg2}(partVec~=i,:);
                D{2} = pinv(X(partVec~=i,:))*data;
            end
            G1 = D{1}*D{1}'/size(D{1},2);
            G2 = D{2}*D{2}'/size(D{2},2);
            G1 = H*G1*H'; % double centered
            G2 = H*G2*H';
            RDM1 = diag(C*G1*C')';
            RDM2 = diag(C*G2*C')';
            c1(i) = corr(rsa_vectorizeIPMfull(G1)',rsa_vectorizeIPMfull(G2)');
            dist1 = corr(RDM1',RDM2');
        end
        c = mean(c1);
        dist = mean(dist1);
    end
    lCKA(reg1,reg2)=c; % as similarity; dissimilarity: 1-c
    lCKA(reg2,reg1)=c;
    corrRDM(reg1,reg2)=dist;
    corrRDM(reg2,reg1)=dist;
end
end
function NN                 = construct_neighbourhood(D)
% constructs the neighbourhood in the given distance matrix
nLayer = size(D,1);
[~,ind]=sort(D,2,'descend');
NN = zeros(nLayer);
for i=1:nLayer
    for j=1:nLayer
        NN(i,j) = find(ind(i,:)==j);
    end
end
NN = NN.*(ones(nLayer)-eye(nLayer));
end
function [accuOrder,countMiss,misplaced] = compareOrder(trueOrder,C)
%function [accuOrder,countMiss] = compareOrder(trueOrder,C)
% compares the order of trueOrder of network with estimated connectivity
nLayer = size(trueOrder,1);
for i=1:nLayer
    [~,ord_true] = sort(trueOrder(i,i+1:end),'ascend');
    [~,ord_est] = sort(C(i,i+1:end),'descend');
    accu(i) = sum(ord_true==ord_est);
    countM(i) = sum(abs(ord_true-ord_est));
    bigMiss(i) = sum(diff(ord_est)~=1);
end
accuOrder = sum(accu)/(nLayer*(nLayer-1)/2);
countMiss = sum(countM);
misplaced = sum(bigMiss);
end
function wCos=cosineW(A,Sig) % Weighted cosine similarity measure 
    % A: N x q vector 
    % Sig: qxq variance matrix 
    % Output: 
    % N*N connectivity matrix of weighted inner product of RDMs (cosineW) 
    C = indicatorMatrix('allpairs',1:size(A,1));
    wCos = zeros(size(A,1));
    [V,L]=eig(Sig);
    l=diag(L);
    sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
    for i=1:size(C,1)
        ind1 = C(i,:)==1;
        ind2 = C(i,:)==-1;
        w1 = A(ind1,:)*sq;
        w2 = A(ind2,:)*sq;
        w1=bsxfun(@rdivide,w1,sqrt(sum(w1.^2,2)));
        w2=bsxfun(@rdivide,w2,sqrt(sum(w2.^2,2)));
        r=w1*w2';
        wCos(ind1,ind2) = r; wCos(ind2,ind1) = r;
    end

end