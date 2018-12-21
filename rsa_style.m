function [canvas,colours,opt] = rsa_style(styID)
%% Description
%   Default plotting parameters specified by the style styID
%   Define additional styles within the switch statement. Users can create
%   their own styles for their projects and use the style package to point
%   towards the user-defined style
%
%
% Author
%   Naveed Ejaz (ejaz.naveed@gmail.com)

canvas              = 'blackonwhite';
opt                 = [];
opt.save.journal    = 'brain';

switch(styID)
    case 'default'
        colours                 = {'blue','green','red','orange','aqua','magenta','yellow','black'};
        opt.general.linestyle   = {'-','-','-','-','-','-','-','-',...
                                   '--','--','--','--','--','--','--','--'};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.legend.leglocation  = 'eastoutside';
    case 'gray'
        colours                 = {'black','lightgray','darkgray','black','lightgray','darkgray'};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v','^'};
        opt.general.linestyle   = {'-','-','-','--','--','--'};
    case 'nomarker'
        colours                 = {'blue','green','red','orange','aqua','magenta','yellow','black'};
        opt.general.linestyle   = {'-','-','-','-','-','-','-','-',...
                                   '--','--','--','--','--','--','--','--'};
        opt.general.markertype  = 'none';
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.legend.leglocation  = 'eastoutside';      
    case 'rsa1'
        % set different shades for colours - colour-blind friendly
        c2=[140 140 185]/255;
        c3=[249 191 193]/255;
        c1=[0 0 200]/255;
        ms=4;
        colours                 = {c1,c1,c1,c2,c2,c2,c3,c3,c3};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v','^'};
        opt.general.linestyle   = {'-','--','-.'};
        opt.general.markersize  = ms;
    case 'rsa2'
        % set different shades for colours - colour-blind friendly
        c2=[140 140 185]/255;
        c3=[249 191 193]/255;
        c1=[0 0 200]/255;
        ms=4;
        colours                 = {c1,c1,c2,c2,c3,c3};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'v','^'};
        opt.general.linestyle   = {'--','-.'};
        opt.general.markersize  = ms;
      case 'rsa3'
        % set different shades for colours - colour-blind friendly
        c2=[140 140 185]/255;
        c3=[249 191 193]/255;
        c1=[0 0 200]/255;
        ms=4;
        colours                 = {c1,c2,c3};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'^'};
        opt.general.linestyle   = {'-.'};
        opt.general.markersize  = ms;      
 

end;