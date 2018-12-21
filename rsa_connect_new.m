function varargout = rsa_connect_new(what,varargin)

baseDir = '/Volumes/MotorControl/data/rsa_connectivity';
codeDir = '/Users/Eva/Documents/MATLAB/projects/rsa_connectivity';
% example RDM distance matrix
%load(fullfile(baseDir,'RDM.mat'));

distLabels={'corr','cosine','euclidean','anzellotti'};

% for plotting
gray=[80 80 80]/255;
lightgray=[160 160 160]/255;
lightlightgray=[200 200 200]/255;
silver=[240 240 240]/255;
black=[0 0 0]/255;
blue=[49,130,189]/255;
lightblue=[158,202,225]/255;
red=[222,45,38]/255;
lightred=[252,146,114]/255;
sAll = style.custom({black,gray,lightlightgray,red,lightred,blue});

c1=[39 38 124]/255;
c2=[140 140 185]/255;
c3=[249 191 193]/255;
c1=[0 0 200]/255;
sBW = style.custom({black,gray,lightgray,silver});
sB  = style.custom({c1,c2,c3});

switch(what)

    case 'run_simulation'
        nCond = 5;
        nPart = 8;
        nVox = 100;
        numSim = 50;
        %varReg = [0,1,10,60];
        varReg = [0,1,5,10:5:30];
        corrReg = [0:0.1:0.9];
        %corrReg = [0,0.1,0.5,0.9];
        type=3; % type 1: G1=G2 ~= G3; type 2: G1~=G2~=G3 (but dist(G2-G1)<dist(G3-G1))
        noiseType='within';
        distCalc = 'ssqrt'; % ssqrt or squared
        vararginoptions(varargin,{'nCond','numSim','RDMtype','corrRDM','type','noiseType','distCalc'});
        NN=[]; 
        
        switch noiseType
            case 'within'
                varReg = [0:0.5:5,6:1:10,12:2:20];
                corrReg = 0;
            case 'between'
                varReg = 80;
                corrReg = [0:0.1:0.9];
            case 'both'
                 varReg  = [0:0.5:5,6:1:10,12:2:20];
                 corrReg = [0:0.1:0.9];
        end
        T.trueDist=zeros(6,1);
      %  while T.trueDist(2)<0.4
            [G,D] = makeGs(nCond,type);
            %  prepare model
            nRDM = size(G,2);
            for i=1:nRDM
                trueRDM(i,:)=rsa_vectorizeRDM(D{i});
                M{i}=makeModel('sameRDM',G{i},nCond);
            end
            % other details for data generation
            S.numPart = nPart;
            S.numVox  = nVox;
            if strcmp(distCalc,'ssqrt') % take ssqrt - square root
                trueRDM=ssqrt(trueRDM);
            end
            [trueDist1,reg1,reg2] = calcDist(trueRDM,'correlation'); % true reg distance - save T
            [trueDist2,reg1,reg2] = calcDist(trueRDM,'cosine');
            T.trueDist=[trueDist1;trueDist2];
     %   end
        T.reg1=repmat(reg1,2,1);
        T.reg2=repmat(reg2,2,1);
        for r=corrReg
            for v=varReg
                for n=1:numSim
                    for i=1:nRDM
                        [data{i},partVec,condVec] = makePatterns('G',M{i}.Gc,'signal',1,'nPart',S.numPart,'nVox',S.numVox); %signal 1, noise 0
                    end
                    % add shared noise across regions
                    data = addSharedNoise(data,v,r);
                    for distType=1:6
                        % splithalf or not
                        % calculate RDMs, distances between them
                        % non-cross, crossval, correlation, cosine
                        [N.calcDist,N.reg1,N.reg2] = calcDistAll(data,partVec,condVec,distType,distCalc);
                        sizeAll=size(N.calcDist,1);
                        N.distType = repmat(distType,sizeAll,1);
                        N.varReg     = repmat(v,sizeAll,1);
                        N.corrReg    = repmat(r,sizeAll,1);
                        N.covReg     = repmat(r*v,sizeAll,1);
                        N.numSim     = repmat(n,sizeAll,1);
                        NN=addstruct(NN,N);
                    end
                end; % number of simulations
                fprintf('%d.',find(v==varReg));
            end; % within-reg
            fprintf('\nDone %d/%d\n',find(r==corrReg),numel(corrReg));
        end; % across-reg
        
        save(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc)),'-struct','NN');
        save(fullfile(baseDir,sprintf('simulation_truth_noise_%s_dist_%s',noiseType,distCalc)),'T','trueRDM');
    
        % lineplot(NN.varReg,NN.calcDist,'subset',NN.reg2==3,'errorfcn','std','split',NN.distType)
        % pivottable(NN.distType,NN.varReg,NN.calcDist,'std','subset',NN.reg2==4 & NN.varReg<6)
        % pivottable(NN.distType,NN.varReg,NN.calcDist,'mean','subset',NN.reg2==4 & NN.varReg<6)
    case 'plot_within'
        noiseType='within';
        distCalc='squared';
        vararginoptions(varargin,{'distCalc'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc)));
        load(fullfile(baseDir,sprintf('simulation_truth_noise_%s_dist_%s',noiseType,distCalc)));
        
        legend = {'RDM-corr-reg2','cRDM-corr-reg2','cRDM-split-corr-reg2',...
            'RDM-corr-reg3','cRDM-corr-reg3','cRDM-split-corr-reg3',...
            'RDM-corr-reg4','cRDM-corr-reg4','cRDM-split-corr-reg4'}
        keyboard;
        D=getrow(D,ismember(D.varReg,[0:1:20]));
        figure
        subplot(311)       
        color     = {c1,c1,c1,c2,c2,c2,c3,c3,c3};
        linestyle   = {'-','--','-.'};
        markertype  = {'o','v','^'};

        lineplot(D.varReg,D.calcDist,'subset',D.distType<4,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_shade','linecolor',color,'linestyle',linestyle,...
            'markersize',4,'markercolor',color,'markerfill',color,'markertype',markertype,...
            'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        drawline(T.trueDist(1),'dir','horz','color',c1);
        drawline(T.trueDist(2),'dir','horz','color',c2);
        drawline(T.trueDist(3),'dir','horz','color',c3);
        title('correlation distances');

        subplot(312)
        color     = {c1,c1,c2,c2,c3,c3};
        linestyle   = {'--','-.'};
        markertype  = {'v','^'};
        legend = {'cRDM-cosine-reg2','cRDM-split-cosine-reg2',...
            'cRDM-cosine-reg3','cRDM-split-cosine-reg3',...
            'cRDM-cosine-reg4','cRDM-split-cosine-reg4'};         
        lineplot(D.varReg,D.calcDist,'subset',D.distType>3&D.distType<6,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_shade','linecolor',color,'linestyle',linestyle,...
                     'markersize',4,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        drawline(T.trueDist(4),'dir','horz','color',c1);
        drawline(T.trueDist(5),'dir','horz','color',c2);
        drawline(T.trueDist(6),'dir','horz','color',c3);
        title('cosine distances');
        
        subplot(313)
        legend = {'multiDepend-reg2','multiDepend-reg3','multiDepend-reg4'};
        
        color     = {c1,c2,c3};
        linestyle   = {'-.'};
        markertype  = {'^'};
        lineplot(D.varReg,D.calcDist,'subset',D.distType==6,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_shade','linecolor',color,'linestyle',linestyle,...
                     'markersize',4,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
         xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        title('multivariate dependence (Anzellotti)');

    case 'calc_confusability'    
        noiseType='within';
        distCalc={'ssqrt','squared'};
        vararginoptions(varargin,{'distCalc','noiseType'});
        % choose style
        legLabel = {'RDM-corr','cRDM-corr','cRDM-cross-corr','cRDM-cos','cRDM-cross-cos','multiDepend'};
        color     = {black,lightgray,lightlightgray,red,lightred,blue};
        linestyle   = {'-','--','-.','--','-.','-.'};
        markertype  = {'o','v','^','v','^','^'};
        switch noiseType
            case 'within'
                metric='varReg';
            case 'between'
                metric='corrReg';
        end
        
        figure
        for c=1:length(distCalc)
            D=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc{c})));
            D=getrow(D,ismember(D.varReg,[0:1:20]));
            CC=[];
            for i=unique(D.(metric))'
                for j=1:max(D.distType)
                    T=getrow(D,D.(metric)==i & D.distType==j);
                    dist1=T.calcDist(T.reg2==2);
                    dist2=T.calcDist(T.reg2==3);
                    dist3=T.calcDist(T.reg2==4);
                    C.confus=1-sum(dist1<dist2 & dist2<dist3)/length(dist1);
                    C.distType=j;
                    C.(metric)=i;
                    CC=addstruct(CC,C);
                end
            end
            lineplot(CC.(metric),CC.confus,'split',CC.distType,'style_thickline',...
                     'linecolor',color,'linestyle',linestyle,...
                     'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legLabel);
            drawline(5/6,'dir','horz','linestyle','--');
            title(sprintf('%s',distCalc{c}));
            xlabel(sprintf('%s region noise',noiseType));
            ylabel('Confusability');
        end
    %    pivottable(D.distType,D.reg2,D.calcDist,'mean','subset',D.varReg<6);
    case 'calc_confusability_both'
         noiseType='both';
        distCalc='ssqrt';
        vararginoptions(varargin,{'distCalc','noiseType'});
        % choose style
        legLabel = {'RDM-corr','cRDM-corr','cRDM-cross-corr','cRDM-cos','cRDM-cross-cos','multiDepend'};
        color     = {black,lightgray,lightlightgray,red,lightred,blue};
        linestyle   = {'-','--','-.','--','-.','-.'};
        markertype  = {'o','v','^','v','^','^'};
        
        TT=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc)));
        T1=getrow(TT,ismember(TT.varReg,[1,20]));
        figure
        uV = unique(T1.varReg);
        CC=[];
        
        for c=uV'
            D=getrow(T1,T1.varReg==c);
            for i=unique(D.corrReg)'
                for j=1:max(D.distType)
                    T=getrow(D,D.corrReg==i & D.distType==j);
                    dist1=T.calcDist(T.reg2==2);
                    dist2=T.calcDist(T.reg2==3);
                    dist3=T.calcDist(T.reg2==4);
                    C.confus=1-sum(dist1<dist2 & dist2<dist3)/length(dist1);
                    C.distType=j;
                    C.corrReg=i;
                    C.varReg=c;
                    CC=addstruct(CC,C);
                end
            end
        end
        figure
            lineplot(CC.corrReg,CC.confus,'split',CC.distType,'subset',CC.varReg==20,'style_thickline',...
                     'linecolor',color,'linestyle',linestyle,...
                     'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legLabel);
            drawline(5/6,'dir','horz','linestyle','--');
            title(sprintf('%s',distCalc{c}));
            xlabel(sprintf('%s region noise',noiseType));
            ylabel('Confusability');
            figure
            lineplot(CC.corrReg,CC.confus,'split',[CC.distType CC.varReg],'style_thickline',...
                     'linecolor',color,'linestyle',linestyle,...
                     'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legLabel);
            drawline(5/6,'dir','horz','linestyle','--');
            title(sprintf('%s',distCalc{c}));
            xlabel(sprintf('%s region noise',noiseType));
            ylabel('Confusability');
    case 'plot_between'
        noiseType='between';
        distCalc='squared';
        vararginoptions(varargin,{'distCalc'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc)));
        load(fullfile(baseDir,sprintf('simulation_truth_noise_%s_dist_%s',noiseType,distCalc)));
    
        figure
        subplot(311)
        color     = {c1,c1,c1,c2,c2,c2,c3,c3,c3};
        linestyle   = {'-','--','-.'};
        markertype  = {'o','v','^'};
        legend = {'RDM-corr-reg2','cRDM-corr-reg2','cRDM-split-corr-reg2',...
            'RDM-corr-reg3','cRDM-corr-reg3','cRDM-split-corr-reg3',...
            'RDM-corr-reg4','cRDM-corr-reg4','cRDM-split-corr-reg4'};
        lineplot(D.corrReg,D.calcDist,'subset',D.distType<4,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',8,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        drawline(T.trueDist(1),'dir','horz','color',c1);
        drawline(T.trueDist(2),'dir','horz','color',c2);
        drawline(T.trueDist(3),'dir','horz','color',c3);
        title('correlation distances');
        
        subplot(312)
        color     = {c1,c1,c2,c2,c3,c3};
        linestyle   = {'--','-.'};
        markertype  = {'v','^'};
        legend = {'cRDM-cosine-reg2','cRDM-split-cosine-reg2',...
            'cRDM-cosine-reg3','cRDM-split-cosine-reg3',...
            'cRDM-cosine-reg4','cRDM-split-cosine-reg4'};
        lineplot(D.corrReg,D.calcDist,'subset',D.distType>3&D.distType<6,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',8,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        hold on;
        drawline(T.trueDist(4),'dir','horz','color',c1);
        drawline(T.trueDist(5),'dir','horz','color',c2);
        drawline(T.trueDist(6),'dir','horz','color',c3);
        title('cosine distances');
        
        subplot(313)
        color     = {c1,c2,c3};
        linestyle   = {'-.'};
        markertype  = {'^'};
        legend = {'Anzellotti-reg2','Anzellotti-reg3','Anzellotti-reg4'};
        lineplot(D.corrReg,D.calcDist,'subset',D.distType==6,'errorfcn','std',...
            'split',[D.reg2 D.distType ],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',8,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color,'leg',legend,'leglocation','southeast');
        xlabel('within region noise');
        ylabel('estimated dist (level 2)');
        title('multivariate dependence (Anzellotti)');
    case 'plot_between_multi'
        noiseType='both';
        distCalc='ssqrt';
        vararginoptions(varargin,{'distCalc'});
        % choose style
        
        D=load(fullfile(baseDir,sprintf('simulation_dist_noise_%s_dist_%s', noiseType,distCalc)));
        load(fullfile(baseDir,sprintf('simulation_truth_noise_%s_dist_%s',noiseType,distCalc)));
    
        T = getrow(D,ismember(D.varReg,[1,20]));
        color     = {black,lightlightgray,red,lightred,blue};
        linestyle   = {'-','-','-','-','-','--','--','--','--','--','-.','-.','-.','-.','-.'};
        markertype  = {'o','o','o','o','o','v','v','v','v','v','^','^','^','^','^'};
        
        figure
        lineplot(T.corrReg,T.calcDist,'subset',T.reg2==3 & ismember(T.distType,[1,3,4,5,6]),...
            'split',[T.varReg T.distType],'style_thickline','linecolor',color,'linestyle',linestyle,...
                     'markersize',6,'markercolor',color,'markerfill',color,'markertype',markertype,...
                     'errorcolor',color,'shadecolor',color);
        subplot(132)
        lineplot(T.corrReg,T.calcDist,'subset',T.reg2==3 & T.distType==2,'split',T.varReg,'style_thickline');
        subplot(133)
        lineplot(T.corrReg,T.calcDist,'subset',T.reg2==3 & T.distType==5,'split',T.varReg,'style_thickline');
        keyboard;
    
    case 'evaluate'
        type=3;
        vararginoptions(varargin,{'type'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        load(fullfile(baseDir,sprintf('simulation_truth_type%d_new',type)));
        
        EE=[];
        range = [2,2,1];
        for s=1:2
            uRDM = unique(D.RDMtype(D.splitHalf==s-1));
            for rdm=1:length(uRDM)
                D1 = getrow(D,D.RDMtype==uRDM(rdm) & D.splitHalf==s-1);
                for i=unique(D1.distType)' % each distType
                    for c1=unique(D1.corrReg)' % correlation
                        for c2=unique(D1.varReg)' % variance
                            % determine true distance
                            if i==1
                                trueDist(1)=T.trueDist(1);
                                trueDist(2)=T.trueDist(2);
                                trueDist(3)=T.trueDist(3);
                            elseif i==2
                                trueDist(1)=T.trueDist(4);
                                trueDist(2)=T.trueDist(5);
                                trueDist(3)=T.trueDist(6);
                            else
                                trueDist(1)=T.trueDist(1);
                                trueDist(2)=T.trueDist(2);
                                trueDist(3)=1;
                            end
                            Dd = getrow(D1,D1.distType==i & D1.corrReg==c1 & D1.varReg==c2);
                            ind1 = Dd.calcDist(Dd.reg2==2);
                            ind2 = Dd.calcDist(Dd.reg2==3);
                            ind3 = Dd.calcDist(Dd.reg2==4);
                            % obtain metrics - only for distance reg2=2
                            % calculate bias
                            tmpTrue = mean(D1.calcDist(D1.distType==i & D1.corrReg==0 & D1.varReg==c2 & D1.reg2==3));
                            E.bias=abs((mean(ind2)-tmpTrue)/range(i));
                            % calculate variance
                            E.var=std(ind2)/range(i);
                            % determine how often correct structure
                          %  E.confus=sum(ind1<ind2 & ind2 < ind3)/length(ind1);
                          % E.confus=sum(ind2 < ind3)/length(ind1);
                          if type==3
                              E.confus = 1-sum([Dd.calcDist(Dd.reg2==2) < Dd.calcDist(Dd.reg2==3) < Dd.calcDist(Dd.reg2==4)])/length(Dd.calcDist(Dd.reg2==2));
                          elseif type==4
                              E.confus = 1-sum([Dd.calcDist(Dd.reg2==2) < Dd.calcDist(Dd.reg2==4) & Dd.calcDist(Dd.reg2==4) < Dd.calcDist(Dd.reg2==3)])/length(Dd.calcDist(Dd.reg2==2));
                          end
                            % other info          
                            E.varReg    = c2;
                            E.corrReg   = c1;
                            E.distType  = i;
                            E.splitHalf = s-1;
                            E.RDMtype   = uRDM(rdm);
                            EE=addstruct(EE,E);
                        end
                    end
                end
            end
        end
        save(fullfile(baseDir,sprintf('evaluation_type%d_new',type)),'-struct','EE');
    case 'evaluate_tmp'
        D = load(fullfile(baseDir,'sim_tmp'));
        load(fullfile(baseDir,sprintf('simulation_truth_type3_new')));
        EE=[];
        range = [2,2,1];
        type=unique(D.ind)';
        for i=1:max(type)
            D1=getrow(D,D.ind==i);
            for c1=unique(D1.corrReg)' % correlation
                for c2=unique(D1.varReg)' % variance
                    % determine true distance
                    if i==1 || i==2 || i==6
                        trueDist(1)=T.trueDist(1);
                        trueDist(2)=T.trueDist(2);
                        trueDist(3)=T.trueDist(3);
                        range2=range(1);
                    elseif i==3 || i==4
                        trueDist(1)=T.trueDist(4);
                        trueDist(2)=T.trueDist(5);
                        trueDist(3)=T.trueDist(6);
                        range2=range(2);
                    else
                        trueDist(1)=T.trueDist(1);
                        trueDist(2)=T.trueDist(2);
                        trueDist(3)=1;
                        range2=range(3);
                    end
                    Dd = getrow(D1,D1.corrReg==c1 & D1.varReg==c2);
                    ind1 = Dd.calcDist(Dd.reg2==2);
                    ind2 = Dd.calcDist(Dd.reg2==3);
                    ind3 = Dd.calcDist(Dd.reg2==4);
                    % obtain metrics - only for distance reg2=2
                    % calculate bias
                    tmpTrue = mean(D1.calcDist(D1.ind==i & D1.corrReg==0 & D1.varReg==c2 & D1.reg2==3));
                    E.bias=abs((mean(ind2)-trueDist(2))/range2);
                    % calculate variance
                    % determine how often correct structure
                    %  E.confus=sum(ind1<ind2 & ind2 < ind3)/length(ind1);
                    % E.confus=sum(ind2 < ind3)/length(ind1);
                    E.confus = 1-sum([Dd.calcDist(Dd.reg2==2) < Dd.calcDist(Dd.reg2==3) < Dd.calcDist(Dd.reg2==4)])/length(Dd.calcDist(Dd.reg2==2));
                    % other info
                    E.varReg    = c2;
                    E.corrReg   = c1;
                    E.ind       = i;
                    EE=addstruct(EE,E);
                end
                
            end
        end
        save(fullfile(baseDir,'eval_tmp'),'-struct','EE');
    case 'plot_withinNoise'
        type=1;
        splitHalf=0;
        distType=1;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        
        T1 = getrow(D,D.corrReg==0 & D.splitHalf==splitHalf & D.distType==distType);
        figure
        plt.scatter(T1.varReg,T1.calcDist,'subset',T1.reg1==1&T1.reg2==3,'split',T1.RDMtype,'style',sBW);
      % plt.scatter(T1.varReg,T1.calcDist,'subset',T1.reg1==1&T1.reg2==3 & T1.RDMtype==1,'style',sBW);
    case 'plot_withinNoise_histogram'
        type=3;
        splitHalf=0;
        distType=1;
        RDMtype=1;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        
        T1 = getrow(D,D.corrReg==0 & D.splitHalf==splitHalf & D.distType==distType & D.RDMtype==RDMtype);
        unVar = unique(T1.varReg);
        for i=unVar'
            figure
            plt.hist(T1.calcDist,'split',T1.reg2,'subset',T1.varReg==i,'style',sB);
            hold on;
            %drawline(mean(T1.calcDist(T1.reg2==2 & T1.varReg==i)),'dir','vert','color',c1);
            %drawline(mean(T1.calcDist(T1.reg2==3 & T1.varReg==i)),'dir','vert','color',c2);
            %drawline(mean(T1.calcDist(T1.reg2==4 & T1.varReg==i)),'dir','vert','color',c3);
        end
    case 'plot_betweenNoise'
        type=1;
        splitHalf=0;
        RDMtype=1;
        vararginoptions(varargin,{'distType','type','split','RDMtype'});

       for r=RDMtype
           D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
         %  D=getrow(D,ismember(D.varReg,[10,30,60,90]) & D.splitHalf==split & D.RDMtype==RDMtype & D.reg1==2 & D.reg2==3);
           D=getrow(D, D.splitHalf==splitHalf & D.RDMtype==RDMtype & D.reg1==1 & D.reg2==4);
           figure
           for v=1:2 % for each distance
               subplot(1,3,v)
               plt.line(D.corrReg,D.calcDist,'split',D.varReg,'style',sBW,'subset',D.distType==v,'style',sBW);
               xlabel('Shared noise');
               ylabel('Estimated distance');
           end
       end
    case 'plot_betweenNoise_histogram'
        type=3;
        splitHalf=0;
        distType=1;
        RDMtype=2;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('simulation_dist_type%d_new',type)));
        
        T1 = getrow(D,D.varReg==10 & D.splitHalf==splitHalf & D.distType==distType & D.RDMtype==RDMtype);
        cReg = unique(T1.corrReg);
        for i=cReg'
            figure
            plt.hist(T1.calcDist,'split',T1.reg2,'subset',T1.corrReg==i,'style',sB);
            hold on;
            drawline(mean(T1.calcDist(T1.reg2==2 & T1.corrReg==i)),'dir','vert','color',c1);
            drawline(mean(T1.calcDist(T1.reg2==3 & T1.corrReg==i)),'dir','vert','color',c2);
            drawline(mean(T1.calcDist(T1.reg2==4 & T1.corrReg==i)),'dir','vert','color',c3);
        end
    case 'plot_evaluate_within'
        type=3;
        splitHalf=0;
        RDMtype=2;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.corrReg==0 & D.split==splitHalf & (D.RDMtype==RDMtype | D.RDMtype==0));
        figure
        subplot(131)
        plt.line(D.varReg,D.confus,'split',D.distType,'style',sBW);
        ylabel('Confusability');
        xlabel('Within-region noise');
        subplot(132)
        plt.line(D.varReg,D.bias,'split',D.distType,'style',sBW);
        ylabel('Bias');
        xlabel('Within-region noise');
        subplot(133)
        plt.line(D.varReg,D.var,'split',D.distType,'style',sBW);
        ylabel('variance');
        xlabel('Within-region noise');
    case 'plot_evaluate_across_var'
        type=3;
        splitHalf=0;
        RDMtype=2;
        distType=1;
        vararginoptions(varargin,{'type','split','RDMtype','distType'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.split==splitHalf & D.RDMtype==RDMtype & D.distType==distType);
        D=getrow(D,ismember(D.varReg,[10,20,30,60]));
        figure
        subplot(131)
        plt.line(D.corrReg,D.confus,'split',D.varReg,'style',sBW);
        ylabel('Correct structure');
        xlabel('within-region noise');
        subplot(132)
        plt.line(D.corrReg,D.bias,'split',D.varReg,'style',sBW);
        ylabel('Bias');
        xlabel('within-region noise');
        subplot(133)
        plt.line(D.corrReg,D.var,'split',D.varReg,'style',sBW);
        ylabel('variance');
        xlabel('within-region noise');
    case 'plot_evaluate_across_distType'
        type=3;
        splitHalf=1;
        RDMtype=2;
        varReg=10;
        vararginoptions(varargin,{'type','split','RDMtype','distType','varReg'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.split==splitHalf & (D.RDMtype==RDMtype | D.RDMtype==0) & D.varReg==varReg);
       % D=getrow(D,ismember(D.varReg,[10,30,60,90]));
        figure
        subplot(131)
        plt.line(D.corrReg,D.confus,'split',D.distType,'style',sBW);
        ylabel('Confusability');
        xlabel('across-region noise');
        subplot(132)
        plt.line(D.corrReg,D.bias,'split',D.distType,'style',sBW);
        ylabel('Bias');
        xlabel('across-region noise');
        subplot(133)
        plt.line(D.corrReg,D.var,'split',D.distType,'style',sBW);
        ylabel('variance');
        xlabel('across-region noise');
        
       % plt.line(D.corrReg,D.var,'split',D.varReg,'subset',D.distType==2);
    case 'plot_evaluate_across'
        type=3;
        RDMtype=2;
        varReg=60;
        vararginoptions(varargin,{'type','split','RDMtype','distType','varReg'});
        D = load(fullfile(baseDir,sprintf('evaluation_type%d_new',type)));
        
        D=getrow(D,D.RDMtype==RDMtype & D.varReg==varReg);
        figure
        subplot(131)
        plt.line(D.corrReg,D.confus,'split',[D.splitHalf,D.distType],'style',sBW);
        ylabel('Confusability');
        xlabel('within-region noise');
        subplot(132)
        plt.line(D.corrReg,abs(D.bias),'split',[D.splitHalf,D.distType],'style',sBW);
        ylabel('Bias');
        xlabel('within-region noise');
        subplot(133)
        plt.line(D.corrReg,D.var,'split',[D.splitHalf,D.distType],'style',sBW);
        ylabel('variance');
        xlabel('within-region noise');
        
        figure
        t=getrow(D,ismember(D.corrReg,[0.1,0.9]));
        subplot(131)
        plt.bar(t.corrReg,t.confus,'split',[ t.splitHalf t.distType],'style',sBW);
        drawline(0.5,'dir','horz','linestyle','--');
        ylabel('Confusability');
        subplot(132)
        plt.bar(t.corrReg,abs(t.bias),'split',[ t.splitHalf t.distType],'style',sBW);
        drawline(0.5,'dir','horz','linestyle','--');
        ylabel('Bias');
        subplot(133)
        plt.bar(t.corrReg,t.var,'split',[ t.splitHalf t.distType],'style',sBW);
        drawline(0.5,'dir','horz','linestyle','--');
        ylabel('Variance');
        
%         figure
%         subplot(131)
%         plt.bar(t.corrReg,t.confus,'style',sBW);
%         drawline(0.5,'dir','horz','linestyle','--');
%         ylabel('Correct structure');
%         subplot(132)
%         plt.bar(t.corrReg,abs(t.bias),'style',sBW);
%         drawline(0.5,'dir','horz','linestyle','--');
%         ylabel('Bias');
%         subplot(133)
%         plt.bar(t.corrReg,t.var,'split',[ t.split t.distType],'style',sBW);
%         drawline(0.5,'dir','horz','linestyle','--');
%         ylabel('Variance');
        
    case 'run_job'
    %rsa_connect('run_simulation');
    rsa_connect_new('run_simulation','numSim',500,'noiseType','between');
    rsa_connect_new('run_simulation','distCalc','squared','numSim',500,'noiseType','between');
    rsa_connect_new('run_simulation','numSim',500,'noiseType','both');
    rsa_connect_new('run_simulation','distCalc','squared','numSim',500,'noiseType','both');
    
    case 'KL_div'
        % noiseless example
        % different Gs
        % consider cosine, distanceCorr, euclidean, KL
        condN=5;
        theta=1;
        nPart=1;
        nVox=1000;
        figPlot=1;
        vararginoptions(varargin,{'condN','figPlot'});
        
        % first make Gs of interest
        G = makeGs(condN);
        if figPlot
            figure
            for i=1:6
                subplot(3,2,i)
                imagesc(G{i});
            end
        end
        % create a model and data for each G
        for i=1:size(G,2)
            M.Ac = G{i};
            M.numGparams = 1;
            M.theta      = 1;
            M.type       = 'feature';
            
            S.numPart = nPart;
            S.numVox  = nVox;
            [data,partVec,condVec] = pcm_generateData(M,theta,S,1,1,0);
            % remove the mean from the data
            data_rm{i} = bsxfun(@minus,data{1},mean(data{1},1));
            C{i} = cov(data_rm{i}');
            G_data{i} = data_rm{i}*data_rm{i}';
            mu{i} = mean(data_rm{i},2);
            RDM(i,:)=pdist(data_rm{i});
        end
        
        % initialize distance for KL divergence
        D = zeros(size(G,2));
        cosDist = zeros(size(G,2));
        for j=1:size(G,2)
            for k=1:size(G,2)
                D(j,k)=KLdivergence(C{j},C{k},mu{j},mu{k});
                cosDist(j,k)=pdist(RDM([j,k],:),'cosine');
                distCorr(j,k)=rsa_calcDistCorrRDMs(RDM([j,k],:));
            end
        end
        keyboard;
    otherwise
        disp('there is no such case.')
end
end

%  % Local functions

function [G,D] = makeGs(condN,type)
% makes specific Gs
switch type
    case 1 % G1 = G2 ~= G3
        U1 = normrnd(0,1,[condN,6]);
        G{1} = U1*U1';
        G{2} = G{1};
        U3 = normrnd(0,1,[condN,6]);
        G{3} = U3*U3';
    case 2
        U1 = normrnd(0,1,[condN,6]);
        G{1} = U1*U1';
        U3 = normrnd(0,1,[condN,6]);
        G{3} = U3*U3';
        G{2} = G{1}*0.8+G{3}*0.2;
    case 3
        nCond=5;
        H1 = indicatorMatrix('allpairs',[1:5]);  
        H2 = eye(nCond) - 1/nCond;
        D{1}=zeros(nCond); D{2}=D{1};
        D{1}(1,1:5)=1;
        D{1}(2:5,1)=1;
        D{1}(1:nCond+1:end)=0; %
        D{4}(5,1:5)=1;
        D{4}(1:5,5)=1;
        D{4}(1:nCond+1:end)=0;
        D{4}(1:nCond+1:end)=0;
        D{2}=D{1};
        U3 = normrnd(0,1,[5,6]);
        D3 = U3*U3';
        D3(1:5+1:end)=0;
        D{3}=0.2*D3+0.8*(D{1}*0.75+D{4}*0.25);
        
        for i=1:4
            G{i} = -0.5*H2*D{i}*H2';
            G{i} = pcm_makePD(G{i});
            G{i} = G{i}./trace(G{i});
            G{i}(find(isnan(G{i})))=0; % in case of nans
            % recalc D
            D{i}=rsa_squareRDM(diag(H1*G{i}*H1')');
        end
    case 4
        nCond=5;
        D{1}=zeros(nCond); D{2}=D{1};
        D{1}(1:2,1:5)=1;
        D{1}(3:5,1:2)=1;
        D{1}(1:nCond+1:end)=0; %
        tmp(3:5,3:5)=1;
        tmp(1:nCond+1:end)=0;
        tmp(1:nCond+1:end)=0;
        D{2}=D{1};
        U3 = normrnd(0,1,[5,6]);
        D3 = U3*U3';
        D3(1:5+1:end)=0;
        D{3}=0.2*D3+0.5*(D{1}*0.9+tmp*0.1);
        D{4}=0.2*D3+0.9*(D{1}*0.9+tmp*0.1);
        H = eye(nCond) - 1/nCond;
        for i=1:4
            G{i} = -0.5*H*D{i}*H';
            G{i} = G{i}./trace(G{i});
            G{i}(find(isnan(G{i})))=0; % in case of nans
        end
    case 5
        nCond=5;
        H1 = indicatorMatrix('allpairs',[1:5]);  
        H2 = eye(nCond) - 1/nCond;
        D{1}=ones(nCond);
        D{1}(1:nCond+1:end)=0;
        D{2}=D{1};
        D{4}=zeros(nCond);
        D{4}(1,1:5)=1;
        D{4}(2:5,1)=1;
        D{4}(1:nCond+1:end)=0; %
        D{3}=0.5*D{1}+0.5*D{4};
        for i=1:4
            G{i} = -0.5*H2*D{i}*H2';
            G{i} = pcm_makePD(G{i});
            G{i} = G{i}./trace(G{i});
            G{i}(find(isnan(G{i})))=0; % in case of nans
            % recalc D
            D{i}=rsa_squareRDM(diag(H1*G{i}*H1')');
        end
        
end

end
function M = makeModel(rdmType,D,nCond)
M.type       = 'feature';
%M.numGparams = 1;
M.numGparams = 1;
M.theta      = 1; 
switch rdmType
    case 'sameRDM'
        M.Gc=D;
    case 'randomRDM'
        M.Ac=rand(nCond);
        M.Gc = M.Ac*M.Ac';
    case 'combRDM'
        M.Gc=D;
end
end
function data = addSharedNoise(data,var,r)
% input: 
% data - datasets
% alpha - variance
% r - correlation
    nDataset = size(data,2);
    nVox = size(data{1},2);
    nTrials = size(data{1},1);
    Z = normrnd(0,1,nTrials,nDataset*nVox);
    A = eye(nDataset)*var+(ones(nDataset)-eye(nDataset))*var*r;
    P = kron(A,eye(nVox)); 
    Zn = Z*sqrtm(P);     % shared noise matrix across reg
    for i=1:nDataset
        data{i} = data{i} + Zn(:,(i-1)*nVox+1:i*nVox);
    end
end

function [dist,ind1,ind2] = calcDistAll(data,partVec,condVec,distType,distCalc)
dist = {'correlation','cosine'};

switch distType
    case 1 % correlation non-cv
        rdm = makeRDM(data,partVec,condVec,'sqEuc');
        if strcmp(distCalc,'ssqrt') % take ssqrt - square root 
            rdm=ssqrt(rdm);
        end
        [dist,ind1,ind2] = calcDist(rdm,'correlation');
    case 2 % correlation cv
        rdm = makeRDM(data,partVec,condVec,'crossnobis');
        if strcmp(distCalc,'ssqrt') % take ssqrt - square root 
            rdm=ssqrt(rdm);
        end
        [dist,ind1,ind2] = calcDist(rdm,'correlation');
    case 3 % correlation double cv
        rdm = makeRDM_splithalf(data,partVec,condVec,'crossnobis');
        if strcmp(distCalc,'ssqrt') % take ssqrt - square root 
            rdm=ssqrt(rdm);
        end
        [dist,ind1,ind2] = calcDist_splithalf(rdm,'correlation');
    case 4 % cosine single cv
        rdm = makeRDM(data,partVec,condVec,'crossnobis');
        if strcmp(distCalc,'ssqrt') % take ssqrt - square root 
            rdm=ssqrt(rdm);
        end
        [dist,ind1,ind2] = calcDist(rdm,'cosine');
    case 5 % cosine double cv
        rdm = makeRDM_splithalf(data,partVec,condVec,'crossnobis');
        if strcmp(distCalc,'ssqrt') % take ssqrt - square root 
            rdm=ssqrt(rdm);
        end
        [dist,ind1,ind2] = calcDist_splithalf(rdm,'cosine');
    case 6 % anzellotti
        [dist,ind1,ind2] = anzellottiDist(data,partVec,condVec);
end
end

function rdm   = makeRDM(data,partVec,condVec,type)
% function rdm   = makeRDM(data,partVec,condVec)
% makes RDM matrix with given type
nData=size(data,2);
X = indicatorMatrix('identity_p',condVec);
H=indicatorMatrix('allpairs',unique(condVec)');
for st=1:nData
    nVox = size(data{st},2); 
    switch type
        case 'sqEuc' % squared euclidean
            % estimate mean condition pattern per dataset
            D{st}=pinv(X)*data{st};
            % remove mean pattern
            for i=1:length(unique(partVec))
                d(:,:,i)=data{st}(partVec==i,:)*data{st}(partVec==i,:)';
            end
         %   for i=1:length(unique(partVec))
         %       data_2{st}(partVec==i,:)=bsxfun(@minus,data{st}(partVec==i,:),mean(data{st}(partVec==i,:)));
         %   end
           % D_2{st}=pinv(X)*data_2{st};
            % make RDM from non-crossval G
            G=D{st}*D{st}';
          %  G_2=D_2{st}*D_2{st}';
           rdm(st,:)= diag(H*G*H')/nVox;
          %  rdm_2(st,:)= diag(H*G_2*H');
          % rdm(st,:) = pdist(D{st},'euclidean');
        case 'crossnobis' % Squared crossvalidated Euclidean (crossnobis)
            % calculate crossvalidated squared Euclidean distances
            rdm(st,:)=rsa.distanceLDC(data{st},partVec,condVec);
        case 'correlation' % correlation
            % estimate mean condition pattern per dataset
            D{st}=pinv(X)*data{st};
            D{st}=bsxfun(@minus,D{st},mean(D{st},1));
            G=D{st}*D{st}';
            % calculate correlation distance (mean subtracted)
            rdm(st,:)=corr_crossval(G,'reg','minvalue');
        case 'cosine' % cosine
            % estimate mean condition pattern per dataset
            D{st}=pinv(X)*data{st};
            G=D{st}*D{st}';
            % calculate cosine distance (mean not subtracted)
            rdm(st,:)=corr_crossval(G,'reg','minvalue');
    end
end;

end
function rdm   = makeRDM_splithalf(data,partVec,condVec,type)
% function rdm   = makeRDM_splithalf(data,partVec,condVec)
% makes crossvalidated RDM matrix for even / odd split
nData=size(data,2);
X = indicatorMatrix('identity_p',condVec);
H=indicatorMatrix('allpairs',unique(condVec)');
% split even and odd runs
idx(:,1) = mod(partVec,2)==1;
idx(:,2) = mod(partVec,2)==0;
count=1;

for st=1:nData
    nVox = size(data{st},2); 
    for p=1:2 % partition
        switch type
            case 'sqEuc' % squared euclidean
                % estimate mean condition pattern per dataset
                D=pinv(X(idx(:,p),:))*data{st}(idx(:,p),:);               
                % make RDM from non-crossval G
                G=D*D';
                rdm(count,:)= diag(H*G*H')/nVox;
               %rdm(count,:) = pdist(D,'euclidean');
            case 'crossnobis' % crossvalidated squared euclidean (crossnobis)
                % calculate crossvalidated squared Euclidean distances
                rdm(count,:)=rsa.distanceLDC(data{st}(idx(:,p),:),partVec(idx(:,p)),condVec(idx(:,p)));
            case 'correlation'
                % estimate mean condition pattern per dataset
                D=pinv(X(idx(:,p),:))*data{st}(idx(:,p),:);
                D=bsxfun(@minus,D,mean(D,1));
                G=D*D';
                % calculate correlation distance (mean subtracted)
                rdm(count,:)=corr_crossval(G,'reg','minvalue');
            case 'cosine'
                D=pinv(X(idx(:,p),:))*data{st}(idx(:,p),:);
                G=D*D';
                % calculate cosine distance (mean not subtracted)
                rdm(count,:)=corr_crossval(G,'reg','minvalue');
        end
        count = count+1;
    end;  % partition
end; % data
end
function [dist,ind1,ind2] = calcDist(rdm,distType)
% calculate distance metric from the input
% N datasets, D distances
% input: N x D matrix
% dist types:
% 1) correlation
% 2) cosine
% 3) euclidean
% output: dist - distances
% ind1: indicator which rdm taken as first
% ind2: indicator which rdm taken as second
switch (distType)
    %tmp=corr(rdm');
    %dist = (1-tmp(1,[2:4]))';
    case 'correlation' 
        tmp=pdist(rdm,'correlation');
        dist=tmp(1:3)';
    case 'cosine'
        tmp=pdist(rdm,'cosine');
        dist = tmp(1:3)';
end
ind1 = ones(size(dist,1),1);
ind2 = [2:4]';
%tmp=pdist(rdm,'correlation');
%dist = [dist;tmp(1:3)'];

%dist = [dist;(1-[tmp(1,[2,3]) tmp(3,4)])'];

%dist = [dist;[tmp(1) tmp(2) tmp(6)]'];
end

function [dist,ind1,ind2] = calcDist_splithalf(rdm,distType)
% calculate distance metric from the input
% crossvalidated version - within and between dataset
% N datasets, D distances
% input: N x D matrix
% dist types:
% 1) correlation
% 2) cosine
% 3) euclidean
count=1;
ind=ones(3,1);
tmp  = rsa_squareRDM(pdist(rdm,distType));
for c=1:size(ind,1) % for every pair
    if c==1
        dist(count,:) = mean([tmp(1,4) tmp(2,3)]);
    elseif c==2
        dist(count,:) = mean([tmp(1,6) tmp(2,5)]);
    else
        dist(count,:) = mean([tmp(1,8) tmp(2,7)]);
        %dist(count,:) = mean([tmp(5,8) tmp(6,7)]);
    end
    count=count+1;
end
ind1 = ones(3,1);
ind2 = [2:4]';
end

function [dist,ind1,ind2] = anzellottiDist(data,partVec,condVec)
% function [dist,ind1,ind2,distType] = anzellottiDist(data,partVec,condVec)
% calculates a relationship between data of different regions
% for now the distance output is 1-R2 and 1-r
nData = size(data,2);
ind1 = ones(size(data,2)-1,1);
dist = zeros(size(ind1,1),1);
for i=1:size(ind1,1)
    dist(i) = multiDependVox(data{1},data{i+1},partVec,condVec,'type','reduceAB');
    ind2(i,:)=i+1;
end
end
