function varargout = rsa_connect(what,varargin)

baseDir = '/Users/eberlot/Documents/Data/rsa_connectivity';
% example RDM distance matrix
load(fullfile(baseDir,'RDM.mat'));

switch(what)
    
    case 'simulateData'
        numCond     = 5;
        numVox      = 1000;
        numPart     = 8;
        numDataset  = 2;
        numSim      = 1000;
        signal      = 1;
        theta       = 1;
        noise       = [0:0.1:1];
        simuType    = 'random'; % what type of simulation: random, sameRDM
        rdmType     = {'allPart','averagePart','crossval'}; % what type of rdm calculation
        vararginoptions(varargin,{'theta','numCond','numVox','numPart','numDataset','numSim','rdmType','simuType','noise','signal'});
        
        NN=[]; 
        for n=1:size(noise,2);
            for s=1:numSim
                % first generate random data
                S.numPart = numPart;
                S.numVox  = numVox;
                
                for d=1:numDataset % numDataset - nRegion
                    %% make into local function - calculate M
                    M{d}.type       = 'feature';
                    M{d}.numGparams = 1;
                    switch simuType % what type of RDM to use as a model
                        case 'sameRDM'
                            Dr{d}   = D;
                            V(d,:)  = rsa_vectorizeRDM(Dr{d});
                            M{d}.Ac = Dr{d}; % use the distance matrix provided
                        case 'random'
                            Dr{d} = rand(numCond);
                            V(d,:) = rsa_vectorizeRDM(Dr{d});
                            % set diagonals to 0
                            Dr{d}(eye(size(Dr{d}))==1)=0;
                            M{d}.Ac = Dr{d};
                    end
                    %%
                    [data(d),partVec,condVec] = pcm_generateData(M{d},theta,S,1,signal,noise(n));
                end
                % calculate true correlation from inital distances - use
                % pdist here (euc, corr, cos)
                % do before data gen
                trueCorr(1,:) = rsa_calcCorrRDMs(Dr);                   % use corr
                trueCorr(2,:) = rsa_calcCorrRDMs(Dr,'interceptFix',1);  % corrN
                trueCorr(3,:) = rsa_calcDistCorrRDMs(Dr);
                for rt = 1:length(rdmType);
                    %which type of RDM to calculate - 1st level
                    switch rdmType{rt}
                        case 'allPart'
                            rdm = makeRDM_allPart(data);
                        case 'averagePart'
                            rdm = makeRDM_average(data,condVec);
                        case 'crossval'
                            rdm = makeRDM_crossval(data,partVec,condVec);
                    end
                    % calculate distances of two datasets - 2nd level
                    N.corr(1,:) = rsa_calcCorrRDMs(rdm);                    % rsa correlation - not fixed intercept (corr)
                    N.corr(2,:) = rsa_calcCorrRDMs(rdm,'interceptFix',1);   % rsa correlation - fixed intercept (corrN)
                    N.corr(3,:) = rsa_calcDistCorrRDMs(rdm);                % distance correlation
                    N.corrType  = [1;2;3];
                    N.numSim    = [s;s;s];
                    N.noise     = [noise(n);noise(n);noise(n)];
                    N.trueCorr  = trueCorr;
                    N.rdmType   = [rt;rt;rt];
                    NN=addstruct(NN,N);
                end; % rdmType
            end; % num simulation
            fprintf('Simulation done: \tnoise level: %6.1f \n',noise(n));
        end % noise 
        % save the structure
        save(fullfile(baseDir,sprintf('simulations_%s',simuType)),'-struct','NN');

    case 'RDMcomb'
        numCond = 5;
        theta = 1;
        numPart = 8;
        numVox = 1000;
        numSim = 20;
        signal = 1;
        noise  = [0:0.1:1];
        rdmType = {'allPart','averagePart','crossval'};
        vararginoptions(varargin,{'numCond','numPart','numVox','numSim','theta','corrInit','noise','rdmType'});
        
        % defaults for simulations
        S.numPart = numPart;
        S.numVox  = numVox;
        % set two random distance matrices
        clear D;
        D{1} = rand(numCond);
        D{2} = rand(numCond);
        % set diagonals to 0
        D{1}(eye(size(D{1}))==1)=0;
        D{2}(eye(size(D{2}))==1)=0;
        % weights for calculating the third RDM - combination of 1st, 2nd
        w1 = [1:-0.1:0];
        w2 = [0:0.1:1];
        
        RR = [];        
        for n = 1:size(noise,2);
            for i = 1:size(w1,2)
                D{3} = D{1}.*w1(i) + D{2}.*w2(i);
                for s = 1:numSim
                    %generate data from the three distance matrices
                    for d=1:3
                        M{d}.type                   = 'feature';
                        M{d}.numGparams             = 1;
                        M{d}.Ac                     = D{d}; % use the distance matrix
                        [data(d),partVec,condVec]   = pcm_generateData(M{d},theta,S,1,signal,noise(n));
                    end
                    % calculate true correlation from inital distances
                    trueCorr(1,:) = rsa_calcCorrRDMs(D);                   % use corr
                    trueCorr(2,:) = rsa_calcCorrRDMs(D,'interceptFix',1);  % corrN
                    trueCorr(3,:) = rsa_calcDistCorrRDMs(D);
                    
                    for r = 1:length(rdmType);
                        % estimate RDM from data
                        switch rdmType{r}
                            case 'allPart'
                                rdmEst = makeRDM_allPart(data);
                            case 'averagePart'
                                rdmEst = makeRDM_average(data,condVec);
                            case 'crossval'
                                rdmEst = makeRDM_crossval(data,partVec,condVec);
                        end
                        R.corr(1,:) = rsa_calcCorrRDMs(rdmEst);                    % rsa correlation - not fixed intercept (corr)
                        R.corr(2,:) = rsa_calcCorrRDMs(rdmEst,'interceptFix',1);   % rsa correlation - fixed intercept (corrN)
                        R.corr(3,:) = rsa_calcDistCorrRDMs(rdmEst);                % distance correlation
                        R.corrType  = [1;2;3];
                        R.weight1   = [w1(i);w1(i);w1(i)];
                        R.weight2   = [w2(i);w2(i);w2(i)];
                        R.noise     = [noise(n);noise(n);noise(n)];
                        R.trueCorr  = trueCorr;
                        R.numSim    = [s;s;s];
                        R.rdmType   = [r;r;r];
                        RR=addstruct(RR,R);
                    end; % rdmType
                end;  %simulation
                fprintf('%d.\n',i);
            end; %weighted combination
            fprintf('Simulation done: \tnoise level: %6.1f\n',noise(n));
        end; % noise levels
        
        save(fullfile(baseDir,'simulations_combRDM'),'-struct','RR');
        % save structure RR
        for s=1:2
            if s==1
                xvar = RR.weight1;
                yvar = RR.corr(:,2);
            else
                xvar = RR.weight2;
                yvar = RR.corr(:,3);
            end
            figure
            for r=1:length(rdmType)
                subplot(1,length(rdmType),r)
                plt.scatter(xvar,yvar,'split',RR.corrType,'subset',RR.rdmType==r,'leg',{'rsa-intercept','rsa-noIntercept','dist'},'leglocation','northeast');
            end
            plt.match('y');
        end
        
    case 'PLOT_corrEstimation_hist'
        simuType='sameRDM'; % sameRDM or random, combRDM
        rdmType={'allPart','averagePart','crossval'};
        corrLabels={'corr-rsa-noIntercept','corr-rsa-fixIntercept','corr-dist'};
        noiseLevels=[0,0.5,1];
        vararginoptions(varargin,{'simuType','rdmType','noiseLevels'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s.mat',simuType)));
          for s=1:size(rdmType,2);
           D = getrow(DD,DD.rdmType==s); 
            figure
            for n=1:size(noiseLevels,2)
                d=getrow(D,D.noise==noiseLevels(n));
                subplot(1,size(noiseLevels,2),n)
                tLabel = sprintf('Corr noise-%2.1f %s',noiseLevels(n),rdmType{s});
                hist_corr(d.corr,d.corrType,'legLabels',corrLabels,'titleLabel',tLabel);          
            end
          end
    case 'PLOT_corrEstimation'
        simuType='random';
        rdmType={'allPart','averagePart','crossval'};
        corrType={'rsa-noIntercept','rsa-fixIntercept','dist-corr'};
        noiseLevels=0;
        vararginoptions(varargin,{'simuType','rdmType','noiseLevels','rsa_intercept'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
        
        for n=1:size(noiseLevels,2)
            D = getrow(DD,DD.noise==noiseLevels(n));
            figure
            plt.bar(D.corrType,D.corr,'split',D.rdmType,'leg',rdmType,'leglocation','northeast');
            set(gca,'XTick',[2 7 11],'XTickLabel',corrType);
            ylabel('Estimated correlation');
            title(sprintf('Noise levels %1.1f',noiseLevels(n)));
            figure
            plt.bar(D.corrType,D.trueCorr-D.corr,'split',D.rdmType,'leg',rdmType,'leglocation','northeast');
            set(gca,'XTick',[2 7 11],'XTickLabel',corrType);
            ylabel('Discrepancy: true rdm corr - data estimated corr');
            title(sprintf('Noise levels %1.1f',noiseLevels(n)));
        end
    case 'PLOT_est_vs_trueCorr'
        simuType='sameRDM'; % sameRDM, random or combRDM
        rdmType={'allPart','averagePart','crossval'};
        corrLabels={'corr-rsa-noIntercept','corr-rsa-fixIntercept','corr-dist'};
        vararginoptions(varargin,{'simuType','rdmType','noiseLevels'});
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
        figure
        for s=1:size(rdmType,2);
            D = getrow(DD,DD.rdmType==s);
            if strcmp(simuType,'combRDM')
                D.corr = D.corr(:,2);
                D.trueCorr = D.trueCorr(:,2);
            end
            subplot(1,size(rdmType,2),s)
            plt.scatter(D.noise,D.corr-D.trueCorr,'split',D.corrType,'leg',corrLabels,'leglocation','northeast');
            title(sprintf('Corr %s %s',rdmType{s},simuType));
            drawline(0,'dir','horz');
            xlabel('noise levels'); ylabel('estimated - true correlation');
            plt.match('y');
        end
    case 'PLOT_relationDistType'
        simuType='random';
        rdmType={'allPart','averagePart','crossval'};
        rsa_intercept=0; % 0 or 1 - which of the two metrics to use
        noiseLevels=[0,0.5,1];
        vararginoptions(varargin,{'simuType','rdmType','noiseLevels','rsa_intercept'});
        
        if rsa_intercept==0
            corrType=1;
        else
            corrType=2;
        end
        
        DD = load(fullfile(baseDir,sprintf('simulations_%s',simuType)));
          for s=1:size(rdmType,2);
           D = getrow(DD,DD.rdmType==s); 
           if strcmp(simuType,'combRDM')
                D.corr = D.corr(:,2);
            end 
           figure
           for n=1:size(noiseLevels,2)
               d=getrow(D,D.noise==noiseLevels(n));
               subplot(1,size(noiseLevels,2),n)
               %  plt.scatter(d.corr(d.corrType==1),d.corr(d.corrType==2));
               plt.scatter(d.corr(d.corrType==corrType),d.corr(d.corrType==3));
               xlabel('corr-rsa'); ylabel('corr-dist');
               title(sprintf('Corr noise-%2.1f %s %s',noiseLevels(n),rdmType{s},simuType));
               ylim([-0.1 1]);
               drawline(0,'dir','horz');
               drawline(0,'dir','vert');
           end
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
     
    otherwise
        disp('there is no such case.')
end
end

%  % Local functions
function rdm   = makeRDM_average(data,condVec)
% function rdm = makeRDM_average(data,condVec)
% makes RDM matrix from euclidean distances (not crossvalidated)
    numDataset=size(data,2);
    % make indicator matrix for estimating mean pattern
    X = indicatorMatrix('identity_p',condVec);
    for st=1:numDataset
        % estimate mean condition pattern per dataset
        D{st}=pinv(X)*data{st};
        % make RDM from Euclidean distances
        rdm{st}=rsa_squareRDM(pdist(D{st}));
    end; 
end
function rdm   = makeRDM_allPart(data)
% function rdm = makeRDM_allPart(data)
% makes RDM matrix from euclidean distances across all runs (not averaging)
    numDataset=size(data,2);
    % make indicator matrix for estimating mean pattern 
    % one factor for each condition & partition
    X = indicatorMatrix('identity_p',1:size(data{1},1));
    for st=1:numDataset
        % estimate mean condition pattern per dataset
        D{st}=pinv(X)*data{st};
        % make RDM from Euclidean distances
        rdm{st}=rsa_squareRDM(pdist(D{st}));
    end;
end
function rdm   = makeRDM_crossval(data,partVec,condVec)
% function rdm   = makeRDM_crossval(data,partVec,condVec)
% makes crossvalidated RDM matrix
numDataset=size(data,2);
for st=1:numDataset
    % calculate crossvalidated squared Euclidean distances
    rdm{st}=rsa_squareRDM(rsa.distanceLDC(data{st},partVec,condVec));
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
    xlabel('Estimated correlation');
    title(titleLabel);
    
end