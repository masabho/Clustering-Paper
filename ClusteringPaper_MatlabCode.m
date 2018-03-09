function ClusteringPaper_MatlabCode()
% Clustering Paper: Wild Arabiensis with Parity Status Known With Lab Arabiensis
% Author: Masabho Peter
% Created: July 2016
% Modified: March 2018
% Matlab Version: R2016b

%% loading xlsx file with spectra values in matlab
    fileName = 'Workspaces/ClusteringPaper_Data.mat';
    if (exist(fileName, 'file'))
       load(fileName);
    else
       [num,txt,raw] = xlsread('ClusteringPaper_Data.xlsx');
       save(fileName, 'num', 'txt', 'raw');
    end; 

%% Plots a two D plot of clusters for visualization (Fig 1 in the manuscript)
    KMeansUncleaned = num(:,2:2152);
    [cidx2,cmeans2] = kmeans(KMeansUncleaned,2,'dist','sqeuclidean');
    [silh2] = silhouette(KMeansUncleaned,cidx2,'sqeuclidean');

    figure(1);% Fig 1 in the manuscript
     plot(KMeansUncleaned(cidx2==1,1), KMeansUncleaned(cidx2==1,2),'r+','MarkerSize',12);
     hold on
     plot(KMeansUncleaned(cidx2==2,1), KMeansUncleaned(cidx2==2,2),'bo','MarkerSize',12);
     plot(cmeans2(:,1),cmeans2(:,2),'kx','MarkerSize',18,'LineWidth',2)
     xlim([0,4])
     ylim([0,4])
     legend({'Cluster1','Cluster2','Centroids'},'Location','NE','FontSize', 12)
     legend boxoff;
     xlabel('Absorbances at frequency 500 nm','FontSize', 12);
     ylabel('Absorbances at frequency 501 nm','FontSize', 12);
     set(gca, 'xtickLabel', {'0','','','','2','','','','4'},'FontSize', 12)
     set(gca, 'ytickLabel', {'0','','','','2','','','','4'},'FontSize',12)
     set(gca,'fontname','times')
     %print(gcf,'FigureOne_Times_12.tif','-dtiff','-r300');
     %print(gcf,'FigOne.tif','-dtiff','-r300');

%% Cleaning Spectra according to Mayagaya et al. 2009
     matrix = raw;
     cols2remove = [1:499, 2001:2152];
     matrix(:,cols2remove)=[];
     x = matrix;
     
     % convert data from cell format to doubles
     % http://www.mathworks.com/help/matlab/ref/cell2mat.html
     X = cell2mat(x);
     
     % Removing Spectra with absorbance range (Max-Min absorbance)..
     % less than 0.3
     removeMosquitoesBelowThisThreshold = 0.3;
     rowMin = min(X,[],2);
     rowMax = max(X,[],2);
     range = (rowMax - rowMin);
     MinToMaxRange = sort(range);
     indx = (rowMax - rowMin < 0.3);
     X(indx,:) = []; % Independent variables
     
     % Generating Dependent variables 
     classmosq = (raw(:,1));
     classmosq(indx,:) = [];
     WildMosq = classmosq(1:927,:);
     WildMosq = str2double(WildMosq);
     WildMosq = [zeros(927,1),WildMosq];
     LabMosq = classmosq(928:1790,:);
     LabMosq = str2double(LabMosq);
     LabMosq = [ones(863,1),LabMosq];
     ClassMosq = [(WildMosq(:,1));(LabMosq(:,1))];
     

     Spectra_AgeUncontrolled = [ClassMosq X]; % Cleaned Spectra According to Mayagaya
     
%% Computing # of Lab_mosquitoes to imitate an exponential decay curve
    P = 0.83;%Daily survival rate
    n = 25; %Maximum age of in the dataset 
    N = zeros(n,1);
    N(1) = 102; % Number of One day old in the Lab-reared dataset
    for i = 2:n
         N(i)= P*N(i-1);
    end
    N;
    AgeStructure = N;
          
    figure(3)% Fig 3 in the Manuscript (Exponential Decay Bar graph)
    xdata = [1 3 5 7 9 11 15 20 25]; % Age of Lab-Mosquitoes
    NumberLab = [102/306*100 70/306*100 48/306*100 34/306*100 23/306*100 16/306*100 8/306*100 3/306*100 2/306*100];% Percentage Lab
    hb = bar(NumberLab,'BarWidth',1);
    set(hb,'facecolor',[66/256, 191/256, 244/256])
    ylim([0,50])
    ylabel('% Mosquitoes','FontSize', 12);
    xlabel('Age in days','FontSize', 12)
    set(gca,'fontname','times')
    set(gca, 'xtickLabel', {'1','3','5','7','9','11','15','20','25'},'FontSize', 12)
    set(gca, 'ytickLabel', {'0','','','','20','','','','40','',''},'FontSize',12)
    
    % Calls latex commamnd and adds texts on the graph 
     x1 = 0.8;
     y1 = 36;
     txt1 = '$\frac{102}{306}$'
     text(x1, y1, txt1,'Interpreter','latex','Fontsize',16)

     x2 = 1.8;
     y2 = 26;
     txt2 = '$\frac{70}{306}$'
     text(x2, y2, txt2, 'Interpreter','latex','Fontsize',16)

     x3 = 2.7
     y3 = 19
     txt3 = '$\frac{48}{306}$'
     text(x3, y3, txt3, 'Interpreter','latex','Fontsize',16)

     x4 = 3.6
     y4 = 14
     txt4 = '$\frac{34}{306}$'
     text(x4, y4, txt4,'Interpreter','latex','Fontsize',16)

     x5 = 4.7
     y5 = 10
     txt5 = '$\frac{23}{306}$'
     text(x5, y5, txt5,'Interpreter','latex','Fontsize',16)

     x6 = 5.7
     y6 = 8;
     txt6 = '$\frac{16}{306}$'
     text(x6, y6, txt6,'Interpreter','latex','Fontsize',16)

     x7 = 6.8
     y7 = 5
     txt7 = '$\frac{8}{306}$'
     text(x7, y7, txt7,'Interpreter','latex','Fontsize',16)

     x8 = 7.8
     y8 = 3.5
     txt8 = '$\frac{3}{306}$'
     text(x8, y8, txt8,'Interpreter','latex','Fontsize',16)

     x9 = 8.9
     y9 = 3.0
     txt9 = '$\frac{2}{306}$'
     text(x9, y9, txt9,'Interpreter','latex','Fontsize',16)
     %print(gcf,'FigureThree.tif','-dtiff','-r300');              
      
%% Random Selection of Laboratory Reared Mosquitoes to Fit Exponential Decay Curve
     raw(indx,:) = [];
     cols2remove = [2:500, 2002:2152];
     raw(:,cols2remove)=[];
     Labmosq = raw(928:1790,:);
     LabMosq = cell2mat(Labmosq);
     
     OneDayOld = LabMosq(LabMosq==1,:);
     ThreeDay = LabMosq(LabMosq==3,:);
     n = size(ThreeDay,1);
     idx = randsample(n,70);
     ThreeDayOld = ThreeDay(idx,:);
     
     FiveDay = LabMosq(LabMosq==5,:);
     n = size(FiveDay,1);
     idx = randsample(n,48);
     FiveDayOld = FiveDay(idx,:);
     
     SevenDay = LabMosq(LabMosq==7,:);
     n = size(SevenDay,1);
     idx = randsample(n,34);
     SevenDayOld = SevenDay(idx,:);
     
     
     NineDay = LabMosq(LabMosq==9,:);
     n = size(NineDay,1);
     idx = randsample(n,23);
     NineDayOld = NineDay(idx,:);
     
     ElevenDay = LabMosq(LabMosq==11,:);
     n = size(ElevenDay,1);
     idx = randsample(n,16);
     ElevenDayOld = ElevenDay(idx,:);
     
     FifteenDay = LabMosq(LabMosq==15,:);
     n = size(FifteenDay,1);
     idx = randsample(n,8);
     FifteenDayOld = FifteenDay(idx,:);
     
     
     TwentyDay = LabMosq(LabMosq==20,:);
     n = size(TwentyDay,1);
     idx = randsample(n,3);
     TwentyDayOld = TwentyDay(idx,:);
     
     TwentyFiveDay = LabMosq(LabMosq==25,:);
     n = size(TwentyFiveDay,1);
     idx = randsample(n,2);
     TwentyFiveDayOld = TwentyFiveDay(idx,:);
     
     LabMosq_ExponentialDecay = [OneDayOld;ThreeDayOld;FiveDayOld;SevenDayOld;NineDayOld;ElevenDayOld;FifteenDayOld;TwentyDayOld;TwentyFiveDayOld];
     LabMosq_ExponentialDecay = [ones(306,1),LabMosq_ExponentialDecay];
     cols2remove = [2];
     LabMosq_ExponentialDecay(:,cols2remove)=[];
     
     WildMosq_Exponential = Spectra_AgeUncontrolled(1:927,:);
     n = size(WildMosq_Exponential,1);
     idx = randsample(n,306);
     WildMosq_ExponentialDecay = WildMosq_Exponential(idx,:);
     Spectra_ExponentialDecay = [LabMosq_ExponentialDecay;WildMosq_ExponentialDecay];

%% Removing Spectra of Lab-reared Mosquitoes at 3, 5 and 25 days old
     Spectra_Ageremoved = Spectra_AgeUncontrolled;
     rows2remove=[1323:1587];
     Spectra_Ageremoved(rows2remove,:)=[];
     
     K = randperm(1525);
     Randomised_Spectra_Ageremoved =  Spectra_Ageremoved(K(1:1525),:);
     
%% K-Means Clustering on Spectra with age of mosquitoes not controlled
   KMeans_AgeUncontrolled = Spectra_AgeUncontrolled;
   K = randperm(1790);     
   Randomised_KMeans_AgeUncontrolled =  KMeans_AgeUncontrolled(K(1:1790),:); 
   MosqSourceType_KMeans_AgeUncontrolled = Randomised_KMeans_AgeUncontrolled(:,1);
   KMeansSpectra_AgeUncontrolled = Randomised_KMeans_AgeUncontrolled(:,2:1502);
   [cidx2,cmeans2] = kmeans(KMeansSpectra_AgeUncontrolled,2,'dist','sqeuclidean');
   [silh2] = silhouette(KMeansSpectra_AgeUncontrolled,cidx2,'sqeuclidean');
     
    % Box plot of silhouette coeffiecients, Age of Mosquitoes not Controlled
    SilhouetteAgeUncontrolled = [cidx2 silh2];
    sortedSilhouetteAgeUncontrolled = sortrows(SilhouetteAgeUncontrolled,1);
    figure(2)
    subplot(3,2,1); %Fig 2 Panel A in the manuscript
    h = boxplot(sortedSilhouetteAgeUncontrolled(:, 2), sortedSilhouetteAgeUncontrolled(:, 1));
    set(h,{'linew'},{2})
    set(gca,'XTickLabel',{'Cluster one','Cluster two'},'FontSize', 14);
    ylabel('Silhouette coefficient','FontSize', 14)
    set(gca,'fontname','arial')
     
     hold on
     
     subplot(3,2,2)% Fig 2 Panel B in the manuscript
     xdata = [1 2];
     ydata = [498/974*100 476/974*100; 365/816*100 451/816*100];
     hb = bar(xdata,ydata,1);
     set(hb(2),'facecolor',[120/256 198/256 83/256])%got color value using color picker
     ylabel('% Mosquitoes','FontSize', 14);
     set(gca, 'xtickLabel', {'Cluster 1 (N = 974)','Cluster 2 (N = 816)'},'FontSize', 14)
     legend({'Laboratory (N = 863)' 'Wild (N = 927)'}, 'FontSize', 14,...
             'location','NE');
      legend boxoff;
     ylim([0 100]) 
     set(gca,'fontname','arial')
     
      hold on
     
%       % Computing average silhouette value per cluster formed after k-means age
%       % not controlled
%       size(silh2(cidx2==1));% checks the number of silhouette values in cluster one
%       mean(silh2(cidx2==1));% computes mean of silhouette values in cluster one
%       size(silh2(cidx2==2));% checks the number of silhouette values in cluster two
%       mean(silh2(cidx2==2));% computes mean of silhouette values in cluster two
      
      % Analysis Cluster One
      K_AgeUncontrolled_NumberLab_ClusterOne = sum((cidx2==1) & (MosqSourceType_KMeans_AgeUncontrolled == 1))
      K_AgeUncontrolled_NumberWild_ClusterOne = sum((cidx2==1) & (MosqSourceType_KMeans_AgeUncontrolled == 0))
 
      % Analysis Cluster two
      K_AgeUncontrolled_NumberLab_ClusterTwo = sum((cidx2==2) & (MosqSourceType_KMeans_AgeUncontrolled == 1))
      K_AgeUncontrolled_NumberWild_ClusterTwo = sum((cidx2==2) & (MosqSourceType_KMeans_AgeUncontrolled == 0))
      
      % Computing chisquare test
      % Observed data
      n1 = 365; N1 = 451; T1 = 816; % n1 = # lab in cluster 1, N1 = # wild in cluster 1, T1 = total cluster one
      n2 = 498; N2 = 476; T2 = 974; % n2 = # lab in cluster 2, N2 = # wild in cluster 2, T2 = total cluster two
      nT = 863; NT = 927; TT = 1790; % nT = Total lab; NT = total wild; TT = grand total
   
     % Expected counts 
      n1e = (nT * T1)/TT;
      n2e = (nT * T2)/TT;
      N1e = (NT * T1)/TT;
      N2e = (NT * T2)/TT;
       
     % Chi-square test
      observed = [n1 N1 n2 N2];
      expected = [n1e N1e n2e N2e];
      chi_2stat_K_means_AgeUncontrolled = sum((observed-expected).^2 ./ expected)
     
    
%% K-Means Clustering on Age controlled by Exponential Decay Curve
     KMeansExponentialDecay = Spectra_ExponentialDecay;
     K = randperm(612);      
     Randomised_KMeansExponentialDecay = KMeansExponentialDecay(K(1:612),:);
     KMeansExponentialDecay = Randomised_KMeansExponentialDecay(:,2:1502);
     MosqSourceType_KMeans_AgeDecay = Randomised_KMeansExponentialDecay(:,1);
     
     [cidx2,cmeans2] = kmeans(KMeansExponentialDecay,2,'dist','sqeuclidean');
     [silh2] = silhouette(KMeansExponentialDecay,cidx2,'sqeuclidean');
     
    %Box plot of silhouette coeffiecient for cluster quality analysis
    SilhouetteExponentialDecay = [cidx2 silh2];
    sortedSilhouetteExponentialDecay = sortrows(SilhouetteExponentialDecay,1);

    subplot(3,2,3)% Fig 2 Panel C in the manuscript
    h = boxplot(sortedSilhouetteExponentialDecay(:, 2), sortedSilhouetteExponentialDecay(:, 1));
    set(h,{'linew'},{2})
    set(gca,'XTickLabel',{'Cluster one','Cluster two'},'FontSize', 14);
    ylabel('Silhouette coefficient','FontSize', 14)
    set(gca,'fontname','arial')
    
    hold on

    subplot(3,2,4)% Fig 2 Panel D in the manuscript
    xdata = [1 2];
    ydata = [167/327*100 160/327*100; 139/285*100 146/285*100];
    hb = bar(xdata,ydata,1);
    set(hb(2),'facecolor',[120/256 198/256 83/256])
    ylabel('% Mosquitoes','FontSize', 14);
    set(gca, 'xtickLabel', {'Cluster 1 (N = 327)','Cluster 2 (N = 285)'},'FontSize', 14)
    legend({'Laboratory (N = 306)' 'Wild (N = 306)'}, 'FontSize', 14,...
             'location','NE');
      legend boxoff;
     ylim([0 100])
     
     hold on
     
     % Analysis Cluster One
      K_AgeDecay_NumberLab_ClusterOne = sum((cidx2==1) & (MosqSourceType_KMeans_AgeDecay == 1))
      K_AgeDecay_NumberWild_ClusterOne = sum((cidx2==1) & (MosqSourceType_KMeans_AgeDecay == 0))
 
      % Analysis Cluster two
      K_AgeDecay_NumberLab_ClusterTwo = sum((cidx2==2) & (MosqSourceType_KMeans_AgeDecay == 1))
      K_AgeDecay_NumberWild_ClusterTwo = sum((cidx2==2) & (MosqSourceType_KMeans_AgeDecay == 0))
      
      % Computing chisquare test
      % Observed data
      n1 = 167; N1 = 149; T1 = 316; % n1 = # lab in cluster 1, N1 = # wild in cluster 1, T1 = total cluster one
      n2 = 139; N2 = 157; T2 = 296; % n2 = # lab in cluster 2, N2 = # wild in cluster 2, T2 = total cluster two
      nT = 306; NT = 306; TT = 612; % nT = Total lab; NT = total wild; TT = grand total
   
     % Expected counts 
      n1e = (nT * T1)/TT;
      n2e = (nT * T2)/TT;
      N1e = (NT * T1)/TT;
      N2e = (NT * T2)/TT;
       
     % Chi-square test
      observed = [n1 N1 n2 N2];
      expected = [n1e N1e n2e N2e];
      chi2stat_K_meansAgeDecay = sum((observed-expected).^2 ./ expected)
     
   
%% K-Means Clustering on Spectra with age controlled by excluding mosquitoes at 3, 5 and 25 days old    
    KMeansAgeRemoved = Randomised_Spectra_Ageremoved;
    ClusterDataAgeremoved = KMeansAgeRemoved(:,2:1502);
    MosqSourceType_KMeans_AgeRemoved = KMeansAgeRemoved(:,1);
    [cidx2,cmeans2] = kmeans(ClusterDataAgeremoved,2,'dist','sqeuclidean');
    [silh2] = silhouette(ClusterDataAgeremoved,cidx2,'sqeuclidean');
    
    % Plots Box plot of silhouette coeffiecient for cluster quality analysis
     SilhouetteAgeRemoved = [cidx2 silh2];
     sortedSilhouetteAgeRemoved = sortrows(SilhouetteAgeRemoved,1);
     subplot(3,2,5)%Fig 2 Panel E in the manuscript
     h = boxplot(sortedSilhouetteAgeRemoved(:, 2), sortedSilhouetteAgeRemoved(:, 1));
     set(h,{'linew'},{2})
     ylabel('Silhouette coefficient','FontSize', 14)
     set(gca,'XTickLabel',{'Cluster one','Cluster two'},'FontSize', 14);
     set(gca,'fontname','arial')
           
      hold on
      
      subplot(3,2,6)% Fig 2 Panel F in the manuscript
      xdata = [1 2];
      ydata = [337/832*100 495/832*100; 261/693*100 432/693*100];
      hb = bar(xdata,ydata,1);
      set(hb(2),'facecolor',[120/256 198/256 83/256])%got color value using color picker
      ylabel('% Mosquitoes','FontSize', 14);
      set(gca, 'xtickLabel', {'Cluster 1 (N = 832)','Cluster 2 (N = 693)'},'FontSize', 14)
      legend({'Laboratory (N = 598)' 'Wild (N = 927)'}, 'FontSize', 14,...
             'location','NE');
      legend boxoff;
      ylim([0 100])
      set(gca,'fontname','arial')
      
       hold off

       set(gcf, 'Position',[10,10,900, 1500]);
       %print(gcf,'FigureTwo.tif','-dtiff','-r300');
       
       % Analysis Cluster One
      K_AgeRemoved_NumberLab_ClusterOne = sum((cidx2==1) & (MosqSourceType_KMeans_AgeRemoved == 1))
      K_AgeRemoved_NumberWild_ClusterOne = sum((cidx2==1) & (MosqSourceType_KMeans_AgeRemoved == 0))
 
      % Analysis Cluster two
      K_AgeRemoved_NumberLab_ClusterTwo = sum((cidx2==2) & (MosqSourceType_KMeans_AgeRemoved == 1))
      K_AgeRemoved_NumberWild_ClusterTwo = sum((cidx2==2) & (MosqSourceType_KMeans_AgeRemoved == 0))
      
      % Computing chisquare test
      % Observed data
      n1 = 261; N1 = 432; T1 = 693; % n1 = # lab in cluster 1, N1 = # wild in cluster 1, T1 = total cluster one
      n2 = 337; N2 = 495; T2 = 832; % n2 = # lab in cluster 2, N2 = # wild in cluster 2, T2 = total cluster two
      nT = 598; NT = 927; TT = 1525; % nT = Total lab; NT = total wild; TT = grand total
   
     % Expected counts 
      n1e = (nT * T1)/TT;
      n2e = (nT * T2)/TT;
      N1e = (NT * T1)/TT;
      N2e = (NT * T2)/TT;
       
     % Chi-square test
      observed = [n1 N1 n2 N2];
      expected = [n1e N1e n2e N2e];
      chi2stat_K_meansAgeRemoved = sum((observed-expected).^2 ./ expected)
                  
%% K-means Clustering on Beta Coefficients from PLS (K-means approach three)     
    PLS_Spectra = Spectra_AgeUncontrolled;
    K = randperm(1790); 
    PLS_SpectraRandomised = PLS_Spectra(K(1:1790),:);     
    Xtrain = (PLS_SpectraRandomised(:,2:1502));
    Ytrain = (PLS_SpectraRandomised(:,1));
    
    % PLS to generate 10 components
    [XtrainL, Ytrainl, XtrainS, YtrainS, beta] = plsregress(Xtrain, Ytrain, 10);
    
    % Clustering beta coefficients from PLS using K-Means clustering
    [cidx2,cmeans2] = kmeans(XtrainS,2,'dist','sqeuclidean');
    [silh2] = silhouette(XtrainS,cidx2,'sqeuclidean');
      
    % Plots Box plot of silhouette coeffiecient 

     SilhouetteInClusters = [cidx2 silh2];
     sortedSilhouetteInClusters = sort(SilhouetteInClusters);
     figure(4)
     subplot(1,2,1)%S1 Fig Panel A in the Supporting information
     h = boxplot(sortedSilhouetteInClusters(:, 2), sortedSilhouetteInClusters(:, 1));
     set(h,{'linew'},{2})
     ylabel('Silhouette coefficient','FontSize', 14)
     set(gca,'XTickLabel',{'Cluster one','Cluster two'},'FontSize', 14);
     set(gca,'fontname','arial')
     
     hold on

     % Plots a two D plot of clusters

     subplot(1,2,2);%S1 Fig Panel B in the Supporting Information
     plot(XtrainS(cidx2==1,1), XtrainS(cidx2==1,2),'r+','MarkerSize',14);
     hold on
     plot(XtrainS(cidx2==2,1), XtrainS(cidx2==2,2),'bo','MarkerSize',14);
     plot(cmeans2(:,1),cmeans2(:,2),'kx','MarkerSize',18,'LineWidth',2)
     legend({'Cluster 1','Cluster 2','Centroids'},'Location','NE','FontSize', 14)
     legend boxoff;
     xlabel('First PLS component','FontSize', 14);
     ylabel('Second PLS component','FontSize', 14);
     set(gca,'fontname','arial')
     set(gca, 'xtickLabel', {'-0.1','-0.08','-0.06','-0.02','0','0.02','0.04','0.06','0.08',''},'FontSize', 12)
     set(gca, 'ytickLabel', {'-0.1','','-0.06','','-0.02','0','0.02','','0.06',''},'FontSize',14)
     set(gcf, 'Position',[10,10,900, 1500]);
     hold off
    % print(gcf,'FigureS1.tif','-dtiff','-r300');
     
     
%% Hierarchical Clustering on Spectra with Age of Mosquitoes not Controlled
    Hierachy_AgeUncontrolled = Spectra_AgeUncontrolled;
    K = randperm(1790);     
    Randomised_Hierachy_AgeUncontrolled =  Hierachy_AgeUncontrolled(K(1:1790),:); 

     MosqSourceType = Randomised_Hierachy_AgeUncontrolled(:,1);
     DataTocluster = (Randomised_Hierachy_AgeUncontrolled(:,2:1502));
     figure(5)
     subplot(3,2,1);%(Fig 5, Panel A in the manuscript)
     NumCluster = 2;
     dist = pdist(DataTocluster, 'euclidean');
     link = linkage(dist, 'average');
     clust = cluster(link, 'maxclust', NumCluster);
     color = link(end-NumCluster+2,3)-eps;
     [H,T,perm] = dendrogram(link, 30, 'colorthreshold', color);
     set(H,'LineWidth',2)
     ylabel('Height of the link','FontSize', 14);
     xlabel('')
     set(gca, 'xtickLabel', {'','','','','','','','','','','Cluster one'...
      ,'','','','','','','','','','','','','','','','Cluster two',...
      '','',''},'FontSize', 14)
      set(gca,'fontname','arial')
      
      hold on
      
      % Bar graph showing distribution of Lab and Wild in Clusters
      subplot(3,2,2);%Fig 5 Panel B in the Manuscript)
      xdata = [1 2];
      ydata = [440/844*100 404/844*100; 423/946*100 523/946*100];
      hb = bar(xdata,ydata,1);%generates a bar graph with color handle 
      set(hb(2),'facecolor',[120/256 198/256 83/256])% Changes color of the bars
      ylim([0  100])
      ylabel('% Mosquitoes','FontSize', 14);
      set(gca, 'xtickLabel', {'Cluster 1 (N = 844)','Cluster 2 (N = 946) '},'FontSize', 14)
      legend({'Laboratory (N = 863)' 'Wild (N = 927)'}, 'FontSize', 14,...
             'location','NE');
      legend boxoff;
      get(gca,'fontname')  
      set(gca,'fontname','arial')
      hold on
      
      % Number of Lab and Wild mosquitoes in each of the 30 nodes of the
      % Tree with Age of mosquitoes not controlled
      n = 30;
      MosquitoesInNodes_AgeNotControlled = zeros(n,2);
      for i = 1:n
    
           MosquitoesInNodes_AgeNotControlled(i,:) = [(sum((T==i) & (MosqSourceType == 1)))...
           (sum((T==i) & (MosqSourceType == 0)))];
      end
      MosquitoesInNodes_AgeNotControlled 

     % Computing chisquare test for a Tree Age not Controlled
     % Observed data
       n1 = 440; N1 = 404; T1 =  844; % n1 = # lab in cluster 1, N1 = # wild in cluster 1, T1 = total cluster one
       n2 = 423; N2 = 523; T2 =  946; % n2 = # lab in cluster 2, N2 = # wild in cluster 2, T2 = total cluster two
       nT = 863; NT = 927; TT = 1790; % nT = Total lab; NT = total wild; TT = grand total
       
       % Expected counts
       n1e = (nT * T1)/TT;
       n2e = (nT * T2)/TT;
       N1e = (NT * T1)/TT;
       N2e = (NT * T2)/TT;
       
       % Chi-square test
       observed = [n1 N1 n2 N2];
       expected = [n1e N1e n2e N2e];
       chi2stat_Hierarchical_AgeUncontrolled = sum((observed-expected).^2 ./ expected)

%% Hierarchical Clustering, Age controlled by Exponential Decay Curve
     
     Hierachy_ExponentialDecay = Spectra_ExponentialDecay;
     K = randperm(612);      
     RandomisedHierachy_ExponentialDecay = Hierachy_ExponentialDecay(K(1:612),:);
     MosqSourceType = RandomisedHierachy_ExponentialDecay(:,1);
     DataTocluster = (RandomisedHierachy_ExponentialDecay(:,2:1502));
     
     subplot(3,2,3);%(Fig 5, Panel C in the manuscript)
     NumCluster = 2;
     dist = pdist(DataTocluster, 'euclidean');
     link = linkage(dist, 'average');
     clust = cluster(link, 'maxclust', NumCluster);
     color = link(end-NumCluster+2,3)-eps;
     [H,T,perm] = dendrogram(link, 30, 'colorthreshold', color);
     set(H,'LineWidth',2)
     ylabel('Height of the link', 'FontSize',14)
     set(gca, 'xtickLabel', {'','','','','','','','','Cluster one','',''...
         ,'','','','','','','','','','','','','','Cluster two','','',...
                   '','',''},'FontSize', 14)
      get(gca,'fontname')
      set(gca,'fontname','arial')

      hold on
      
     % bar plot
     subplot(3,2,4);%(Fig 5, Panel D in the manuscript)
     xdata = [1 2];
     ydata = [281/563*100 282/563*100; 25/49*100 24/49*100];
     hb = bar(xdata,ydata,1);
     ylim([0 100]);
     set(hb(2),'facecolor',[120/256 198/256 83/256])
     ylabel('% Mosquitoes','FontSize', 14);
     set(gca,'xtickLabel', {'Cluster 1 (N = 563)','Cluster 2 (N = 49)'},'FontSize', 14)
     legend({'Laboratory (N = 306)' 'Wild (N = 306)'}, 'FontSize', 14,...
             'location','NE');
      legend boxoff;
      set(gca,'fontname','arial')
      
      hold on
      
      
      % Number of Lab and Wild mosquitoes in each of the 30 nodes of the
      % Tree with Age of mosquitoes controlled by Exponential Decay curve
      
      n = 30;
      MosquitoesInNodes_AgeControlled_Decay = zeros(n,2);
      for i = 1:n
           MosquitoesInNodes_AgeControlled_Decay(i,:) = [(sum((T==i) & (MosqSourceType == 1)))...
           (sum((T==i) & (MosqSourceType == 0)))];
      end
      MosquitoesInNodes_AgeControlled_Decay 
      

     % Computing Chi-square test Hierarchical Age Controlled by Exponential Decay 
     % Observed data
       n1 = 58; N1 = 61; T1 = 119; % n1 = # lab in cluster 1, N1 = # wild in cluster 1, T1 = total cluster one
       n2 = 248; N2 = 245; T2 = 493; % n2 = # lab in cluster 2, N2 = # wild in cluster 2, T2 = total cluster two
       nT = 306; NT = 306; TT = 612; % nT = Total lab; NT = total wild; TT = grand total
       
       % Expected counts
       n1e = (nT * T1)/TT;
       n2e = (nT * T2)/TT;
       N1e = (NT * T1)/TT;
       N2e = (NT * T2)/TT;
       
       % Chi-square test
       observed = [n1 N1 n2 N2];
       expected = [n1e N1e n2e N2e];
       chi2stat_ExponentialDecay = sum((observed-expected).^2 ./ expected)
       
%% Hierarchical Clustering with Lab-mosquitoes at 3, 5, 25 days old excluded
 
      Hierachy_Spectra_Ageremoved = Randomised_Spectra_Ageremoved;
      MosqSourceType = Hierachy_Spectra_Ageremoved(:,1);
      DataTocluster = (Hierachy_Spectra_Ageremoved(:,2:1502));
      subplot(3,2,5);
      NumCluster = 2;
      dist = pdist(DataTocluster, 'euclidean');
      link = linkage(dist, 'average');
      clust = cluster(link, 'maxclust', NumCluster);
      color = link(end-NumCluster+2,3)-eps;
      [H,T,perm] = dendrogram(link, 30, 'colorthreshold', color);
      set(H,'LineWidth',2)
      ylabel('Height of the link', 'FontSize',12)
      set(gca, 'xtickLabel', {'','','','','','','','Cluster one','','',''...
                   ,'','','','','','','','','','','','Cluster two','','','','',...
                   '','',''},'FontSize', 14)
      get(gca,'fontname')  % shows you what you are using.
      set(gca,'fontname','arial') 

       hold on
       % Bar graph after mosquitoes at 3, 5, 25 days old were removed from the analysis
       subplot(3,2,6);
       xdata = [1 2];
       ydata = [132/307*100 175/307*100; 466/1218*100 752/1218*100];
       hb = bar(xdata,ydata,1);
       set(hb(2),'facecolor',[120/256 198/256 83/256])%got color value using color picker
       ylim([0 100]);
       ylabel('% Mosquitoes','FontSize', 14);
       set(gca, 'xtickLabel', {'Cluster 1 (N = 307)','Cluster 2 (N = 1218)'},'FontSize', 14)
       legend({'Laboratory (N = 598)' 'Wild (N = 927)'}, 'FontSize', 14,...
             'location','NE');
       legend boxoff;
       get(gca,'fontname')  % shows you what you are using.
       set(gca,'fontname','arial') 
       
       hold off

       set(gcf, 'Position',[10,10,900, 1500]); % specify figure size in pixels
       %print(gcf,'FigureFive.tif','-dtiff','-r300');
       
       % Number of Lab and Wild mosquitoes in each of the 30 nodes of the
       % Tree with Age of mosquitoes controlled by excluding lab Mosq at
       % 3,5,25 days old
       n = 30;
       MosquitoesInNodes_AgeControlled_Remove = zeros(n,2);
       for i = 1:n
    
           MosquitoesInNodes_AgeControlled_Remove(i,:) = [(sum((T==i) & (MosqSourceType == 1)))...
           (sum((T==i) & (MosqSourceType == 0)))];
       end
       MosquitoesInNodes_AgeControlled_Remove 


    % Computing Chi-square test for a Tree Age Controlled by Removing
    % Observed data
       n1 = 132; N1 = 175; T1 =  307; % n1 = # lab in cluster 1, N1 = # wild in cluster 1, T1 = total cluster one
       n2 = 466; N2 = 752; T2 = 1218; % n2 = # lab in cluster 2, N2 = # wild in cluster 2, T2 = total cluster two
       nT = 598; NT = 927; TT = 1525; % nT = Total lab; NT = total wild; TT = grand total
       
       % Expected counts
       n1e = (nT * T1)/TT;
       n2e = (nT * T2)/TT;
       N1e = (NT * T1)/TT;
       N2e = (NT * T2)/TT;
       
       % Chi-square test
       observed = [n1 N1 n2 N2];
       expected = [n1e N1e n2e N2e];
       chi2stat_Hierachy_AgeRemoved = sum((observed-expected).^2 ./ expected)
 
end
     
