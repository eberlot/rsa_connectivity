function varargout = rsa_connect(what,varargin)

baseDir = '/Users/eberlot/Documents/Data/rsa_connectivity';
% example RDM distance matrix
load(fullfile(baseDir,'RDM.mat'));

averageType={'allPart','averagePart','crossval'};
distLabels={'corr','cosine','euclidean','distcorr'};

switch(what)
    
    case 'simulateData'
        nCond       = 5;
        nVox        = 1000;
        nPart       = 8;
        nRegion     = 2;
        nSim        = 1000;
        signal      = 1;
        theta       = 1;
        noise       = [0:0.1:1];
        rdmType     = 'randomRDM'; % what type of simulation: randomRDM, sameRDM
        vararginoptions(varargin,{'theta','numCond','numVox','numPart','numDataset','numSim','rdmType','simuType','noise','signal'});
        
        NN=[]; 
        for n1=1:size(noise,2);
            for n2=1:size(noise,2)
                for s=1:nSim
                    % 1) generate data
                    S.numPart = nPart;
                    S.numVox  = nVox;
                        clear D;
                    for d=1:nRegion
                        D{d}    = randn(nCond);
                        if strcmp(rdmType,'sameRDM')
                            D{d} = D{1};
                        end
                        M{d}    = makeModel(rdmType,D{d},nCond); % make model for data generation
                        if d==1
                            nIdx = noise(n1);
                        else
                            nIdx = noise(n2);
                        end
                        [data(2),partVec,condVec] = pcm_generateData(M{d},theta,S,1,10,nIdx);
                        trueRDM(d,:)  = rsa_vectorizeRDM(M{d}.Ac); % trueRDM
                    end
                    
                    % 2) calculate true distance
                    N.trueDist = calcDist(trueRDM);
                    
                    % 3) evaluate
                    for rt = 1:length(averageType);
                        % 3a) how to calculate RDM - 1st level
                        switch averageType{rt}
                            case 'allPart'
                                calcRDM = makeRDM_allPart(data);
                            case 'averagePart'
                                calcRDM = makeRDM_average(data,condVec);
                            case 'crossval'
                                calcRDM = makeRDM_crossval(data,partVec,condVec);
                        end
                        % 3b) calculate distances of two datasets - 2nd level
                        N.calcDist = calcDist(calcRDM);
                        % dist type labelling
                        N.distType        = [1:4]';
                        N.distLabel(1,:)  = {'correlation'};
                        N.distLabel(2,:)  = {'cosine'};
                        N.distLabel(3,:)  = {'euclidean'};
                        N.distLabel(4,:)  = {'distCorr'};
                        % other info
                        N.numSim        = repmat(s,4,1);
                        N.noise1        = repmat(noise(n1),4,1);
                        N.noise2        = repmat(noise(n2),4,1);
                        N.averageType   = repmat(rt,4,1);
                        NN=addstruct(NN,N);
                    end; % rdmType
                end; % num simulation
                fprintf('Simulation done: \tnoise level: %2.1f-%2.1f \n',noise(n1),noise(n2));
            end % noise
        end
        % save the structure
        save(fullfile(baseDir,sprintf('simulations_%s',rdmType)),'-struct','NN');

    case 'simulateData_comb'
        nCond = 5;
        theta = 1;
        nPart = 8;
        nVox = 1000;
        nSim = 20;
        signal = 1;
        noise  = [0:0.1:1];
        vararginoptions(varargin,{'numCond','numPart','numVox','numSim','theta','corrInit','noise','rdmType'});
        
        % defaults for simulations
        S.numPart = nPart;
        S.numVox  = nVox;
        % set two random distance matrices
        clear D;
        D{1} = rand(nCond);
        D{2} = rand(nCond);
        % set diagonals to 0
        D{1}(eye(size(D{1}))==1)=0;
        D{2}(eye(size(D{2}))==1)=0;
        % weights for calculating the third RDM - combination of 1st, 2nd
        w1 = [1:-0.1:0];
        w2 = [0:0.1:1];
        
        NN = [];        
        for n = 1:size(noise,2);
            for i = 1:size(w1,2)
                D{3} = D{1}.*w1(i) + D{2}.*w2(i);
                for s = 1:nSim
                    %1) generate data from the three distance matrices
                    for d=1:3
                        M{d}    = makeModel('combRDM',D{d},nCond); % make model for data generation
                        [data(d),partVec,condVec] = pcm_generateData(M{d},theta,S,1,signal,noise(n));
                        trueRDM(d,:)  = rsa_vectorizeRDM(M{d}.Ac); % trueRDM
                    end
                    % 2) calculate true distance
                    N.trueDist = calcDist(trueRDM);
                    
                    for r = 1:length(averageType);
                        % 3) estimate RDM from data
                        % 3a) average data
                        switch averageType{r}
                            case 'allPart'
                                calcRDM = makeRDM_allPart(data);
                            case 'averagePart'
                                calcRDM = makeRDM_average(data,condVec);
                            case 'crossval'
                                calcRDM = makeRDM_crossval(data,partVec,condVec);
                        end
                        
                        % 3b) calculate distances of two datasets - 2nd level
                        N.calcDist = calcDist(calcRDM);
                        % dist type labelling
                        N.distType        = [1:4]';
                        N.distLabel(1,:)  = {'correlation'};
                        N.distLabel(2,:)  = {'cosine'};
                        N.distLabel(3,:)  = {'euclidean'};
                        N.distLabel(4,:)  = {'distCorr'};
                        % other info
                        N.numSim        = repmat(s,4,1);
                        N.noise1        = repmat(noise(n),4,1);
                        N.noise2        = repmat(noise(n),4,1);
                        N.weight1       = repmat(w1(i),4,1);
                        N.weight2       = repmat(w2(i),4,1);
                        N.averageType   = repmat(r,4,1);
                        NN=addstruct(NN,N);
                        
                    end; % rdmType
                end;  %simulation
                fprintf('%d.\n',i);
            end; %weighted combination
            fprintf('Simulation done: \tnoise level: %2.1f\n',noise(n));
        end; % noise levels
        
        save(fullfile(baseDir,'simulations_combRDM'),'-struct','NN');
        
    case 'evaluate'
      rdmType = 'randomRDM'; % what type of simulation: random, sameRDM
      vararginoptions(varargin,{'rdmType'});
      T = load(fullfile(baseDir,sprintf('simulations_%s',rdmType)));
      EE=[];
      for n1=unique(T.noise1)'
          for n2=unique(T.noise2)'
              R1=pivottable(T.averageType,T.distType,[T.trueDist],'mean','subset',T.noise1==n1 & T.noise2==n2);
              R2=pivottable(T.averageType,T.distType,[T.calcDist],'mean','subset',T.noise1==n1 & T.noise2==n2);
              R3=pivottable(T.averageType,T.distType,[T.calcDist-T.trueDist],'mean','subset',T.noise1==n1 & T.noise2==n2);
              R4=pivottable(T.averageType,T.distType,[T.calcDist],'var','subset',T.noise1==n1 & T.noise2==n2);
              R5=pivottable(T.averageType,T.distType,[T.trueDist-T.calcDist],'sqrt(mean(x.^2))','subset',T.noise1==n1 & T.noise2==n2);
              
              E.trueDist    = R1(:);
              E.calcDist    = R2(:);
              E.bias        = R3(:);
              E.var         = R4(:);
              E.rmse        = R5(:);
              E.averageType = repmat([1:3]',4,1);
              E.distType    = kron([1:4],ones(1,3))';
              E.noise1      = repmat(n1,12,1);
              E.noise2      = repmat(n2,12,1);

              EE=addstruct(EE,E);
          end
      end
      
      save(fullfile(baseDir,sprintf('evaluate_%s',rdmType)),'-struct','EE');
   
    case 'noiseless'    
        nCond = 5;      
        signal = [1,5,10:10:100];
        theta = 1;
        noise = 0;
        nPart = 9;
        nVox = 1000;
        vararginoptions(varargin,{'nCond','signal','theta','noise'});

        D = randn(nCond);
        NN=[]; 
        S.numPart = nPart;
        S.numVox  = nVox;
        trueRDM=repmat(rsa_vectorizeRDM(D),2,1);
        trueDist = calcDist(trueRDM);
        M=makeModel('sameRDM',D,nCond); 
        for s=signal
            [data(1),partVec,condVec] = pcm_generateData(M,theta,S,1,1,noise);
            [data(2),partVec,condVec] = pcm_generateData(M,theta,S,1,s,noise);
            N.calcDist = calcDist(trueRDM);
            calcRDM = makeRDM_crossval(data,partVec,condVec);
            N.calcDist = calcDist(calcRDM);
            % dist type labelling
            N.distType        = [1:4]';
            N.distLabel(1,:)  = {'correlation'};
            N.distLabel(2,:)  = {'cosine'};
            N.distLabel(3,:)  = {'euclidean'};
            N.distLabel(4,:)  = {'distCorr'};
            N.signal1 = repmat(1,4,1);
            N.signal2 = repmat(s,4,1);
            NN = addstruct(NN,N);
        end

        figure
        plt.line(NN.signal2,NN.calcDist,'split',NN.distType,'leg',{'correlation','cosine','euclidean','distCorr'},'leglocation','northwest');
        ylabel('distance between identical RDMs');
        xlabel('signal proportion of RDM 1 vs. 2');
    case 'noiseDiff'        
        nCond = 5;      
        signal = 1;
        theta = 1;
        noise = [0:0.1:1];
        nPart = 9;
        nVox = 1000;
        vararginoptions(varargin,{'nCond','signal','theta','noise'});

        D = randn(nCond);
        NN=[]; 
        S.numPart = nPart;
        S.numVox  = nVox;
        trueRDM=repmat(rsa_vectorizeRDM(D),2,1);
        trueDist = calcDist(trueRDM);
        M=makeModel('sameRDM',D,nCond); 
        for s=noise
            [data(1),partVec,condVec] = pcm_generateData(M,theta,S,1,1,0);
            [data(2),partVec,condVec] = pcm_generateData(M,theta,S,1,1,s);
            N.calcDist = calcDist(trueRDM);
            calcRDM = makeRDM_crossval(data,partVec,condVec);
            N.calcDist = calcDist(calcRDM);
            % dist type labelling
            N.distType        = [1:4]';
            N.distLabel(1,:)  = {'correlation'};
            N.distLabel(2,:)  = {'cosine'};
            N.distLabel(3,:)  = {'euclidean'};
            N.distLabel(4,:)  = {'distCorr'};
            N.noise1 = repmat(1,4,1);
            N.noise2 = repmat(s,4,1);
            NN = addstruct(NN,N);
        end

        figure
        plt.line(NN.noise2,NN.calcDist,'split',NN.distType,'leg',{'correlation','cosine','euclidean','distCorr'},'leglocation','northwest');
        ylabel('distance between identical RDMs');
        xlabel('noise proportion of RDM 1 vs. 2');



    case 'PLOT_dist_hist'
        simuType='sameRDM'; % sameRDM or randomRDM, combRDM
        noiseLevels=[0,0.5,1];
        noiseType=1; % 1 or 0; 1 - same noise type, 0 opposite
        vararginoptions(varargin,{'simuType','simuType','noiseLevels','noiseType'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s.mat',simuType)));
          for s=1:size(averageType,2);
           D = getrow(DD,DD.averageType==s); 
            figure
            for n=1:size(noiseLevels,2)
                if noiseType==1
                    d=getrow(D,D.noise1==noiseLevels(n) & D.noise2==noiseLevels(n));
                else
                    d=getrow(D,D.noise1==noiseLevels(n) & D.noise1+D.noise2==1);    
                end  
                subplot(1,size(noiseLevels,2),n)
                tLabel = sprintf('Corr noise-%2.1f %s',noiseLevels(n),averageType{s});
                hist_corr(d.calcDist,d.distType,'legLabels',distLabels,'titleLabel',tLabel);          
            end
          end
    case 'PLOT_calc_trueDist'
        simuType='sameRDM'; % sameRDM, randomRDM or combRDM
        noiseType=1; % 1=same or 0=opposite
        vararginoptions(varargin,{'simuType','rdmType','noiseType'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
        figure
        for s=1:size(averageType,2);
            D = getrow(DD,DD.averageType==s);
            if noiseType==1
                D = getrow(D,D.noise1==D.noise2);
            else
                D = getrow(D,D.noise1+D.noise2==1);
            end
            if strcmp(simuType,'combRDM')
                D.calcDist = D.calcDist(:,2);
                D.trueDist = D.trueDist(:,2);
            end
            subplot(1,size(averageType,2),s)
            plt.scatter(D.noise1,D.calcDist-D.trueDist,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title(sprintf('Corr %s %s',averageType{s},simuType));
            drawline(0,'dir','horz');
            xlabel('noise levels'); ylabel('estimated - true distance');
            plt.match('y');
        end 
    case 'PLOT_evaluate'
        simuType = 'randomRDM';
        noiseType=1; % 1=same or 0=opposite
        vararginoptions(varargin,{'simuType','rdmType','noiseType'});
        
        DD = load(fullfile(baseDir,sprintf('evaluate_%s',simuType)));
        for s=1:size(averageType,2);
            figure
            D = getrow(DD,DD.averageType==s);
            if noiseType==1
                D = getrow(D,D.noise1==D.noise2);
            else
                D = getrow(D,D.noise1+D.noise2==1);
            end
            subplot(151)
            plt.scatter(D.noise1,D.trueDist,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title('true distance');
            xlabel('noise levels');
            subplot(152)
            plt.scatter(D.noise1,D.calcDist,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title(sprintf('calculated distance - %s type',averageType{s}));
            subplot(153)
            plt.scatter(D.noise1,D.bias,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            title('bias');
            drawline(0,'dir','horz','linestyle','--');
            subplot(154)
            plt.scatter(D.noise1,D.var,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            drawline(0,'dir','horz','linestyle','--');
            title('variance');
            subplot(155)
            plt.scatter(D.noise1,D.rmse,'split',D.distType,'leg',distLabels,'leglocation','northeast');
            drawline(0,'dir','horz','linestyle','--');
            title('rmse');
        end
        
    case 'PLOT_combRDM'
        noiseLevels=[0,0.5,1];
        rdmType={'allPart','averagePart','crossval'};
        vararginoptions(varargin,{'noiseLevels','rdmType'});
        
        
        DD = load(fullfile(baseDir,'simulations_combRDM'));
        for n=1:length(noiseLevels)
            D = getrow(DD,DD.noise==noiseLevels(n));
            for s=1:2
                if s==1
                    xvar = D.weight1;
                    yvar = D.corr(:,2);
                else
                    xvar = D.weight2;
                    yvar = D.corr(:,3);
                end
                figure
                for r=1:length(rdmType)
                    subplot(1,length(rdmType),r)
                    plt.scatter(xvar,yvar,'split',D.corrType,'subset',D.rdmType==r,'leg',{'rsa-intercept','rsa-noIntercept','dist'},'leglocation','northeast');
                    xlabel('weight for RDM creation');
                    ylabel('corr to RDM with weights on x');
                    title(sprintf('Corr noise-%1.1f %s',noiseLevels(n),rdmType{r}));
                end
                plt.match('y');
            end
        end
    case 'empiricalExample'
       D=load(fullfile(baseDir,'RDMs'));
       regLabels={'ba1','ba2','ba3','ba4','ba6'};
       
       % extract relevant field
       for i=1:length(regLabels)
           % create RDM
           RDM{i}=rsa_squareRDM(getfield(D,regLabels{i}));
       end
       
       corrRSA  = rsa_calcCorrRDMs(RDM);
       corrDist = rsa_calcDistCorrRDMs(RDM);
       
       acrReg_rsa  = rsa_squareRDM(corrRSA);
       acrReg_dist = rsa_squareRDM(corrDist);
       
       figure;
       subplot(1,2,1);
       imagesc(acrReg_rsa,[0 1.5]);
       title('Across regions RSA correlation');
       subplot(1,2,2);
       imagesc(acrReg_dist,[0 1.5]);
       title('Across regions DIST correlation');
     
    case 'run_job'
      % rsa_connect('simulateData');
       %rsa_connect('simulateData','rdmType','sameRDM');
       %rsa_connect('evaluate');
       %rsa_connect('evaluate','rdmType','sameRDM');
       %rsa_connect('simulateData_comb');
       rsa_connect('evaluate','rdmType','combRDM');
       
    otherwise
        disp('there is no such case.')
end
end

%  % Local functions

function M = makeModel(rdmType,D,nCond)
M.type       = 'feature';
M.numGparams = 1;
switch rdmType
    case 'sameRDM'
        M.Ac=D;
    case 'randomRDM'
        M.Ac=rand(nCond);
    case 'combRDM'
        M.Ac=D;
end
end

function dist = calcDist(rdm)
% calculate distance metric from the input
% N datasets, D distances
% input: N x D matrix
% dist types:
% 1) correlation
% 2) cosine
% 3) euclidean
% 4) distance correlation

dist(1,:)  = pdist(rdm,'correlation');
dist(2,:)  = pdist(rdm,'cosine');
dist(3,:)  = pdist(rdm,'euclidean');
dist(4,:) = rsa_calcDistCorrRDMs(rdm);

end

function rdm   = makeRDM_average(data,condVec)
% function rdm = makeRDM_average(data,condVec)
% makes RDM matrix from euclidean distances (not crossvalidated)
    % make indicator matrix for estimating mean pattern
    X = indicatorMatrix('identity_p',condVec);
    for st=1:size(data,2)
        % estimate mean condition pattern per dataset
        D{st}=pinv(X)*data{st};
        % make RDM from Euclidean distances
        rdm(st,:)=pdist(D{st});
    end; 
end
function rdm   = makeRDM_allPart(data)
% function rdm = makeRDM_allPart(data)
% makes RDM matrix from euclidean distances across all runs (not averaging)
    for st=1:size(data,2)
        % make RDM from Euclidean distances
        rdm(st,:)=pdist(data{st});
    end;
end
function rdm   = makeRDM_crossval(data,partVec,condVec)
% function rdm   = makeRDM_crossval(data,partVec,condVec)
% makes crossvalidated RDM matrix
numDataset=size(data,2);
for st=1:numDataset
    % calculate crossvalidated squared Euclidean distances
    rdm(st,:)=rsa.distanceLDC(data{st},partVec,condVec);
end;

end

function hist_corr(corr,corrType,varargin)
% function hist_corr(corr_rsa,corr_dist,vararginoptions)
% plots the distribution of rsa and distance correlations
% varargin - optional level of noise for the title
vararginoptions(varargin,{'legLabels','titleLabel'});

    plt.hist(corr,'split',corrType,'leg',legLabels,'leglocation','northeast');
    drawline(mean(corr(corrType==1)),'dir','vert','color','b');
    drawline(mean(corr(corrType==2)),'dir','vert','color','g');
    drawline(mean(corr(corrType==3)),'dir','vert','color','r');
    drawline(mean(corr(corrType==4)),'dir','vert','color','y');
    xlabel('Estimated distance');
    title(titleLabel);
    
end