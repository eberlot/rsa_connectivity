function [canvas,colours,opt] = DNN_style(styID)
%% Description
%   Default plotting parameters specified by the style styID
%   Define additional styles within the switch statement. Users can create
%   their own styles for their projects and use the style package to point
%   towards the user-defined style
%
%

canvas              = 'blackonwhite';
opt                 = [];
opt.save.journal    = 'brain';

% personalized colours
black = [0 0 0];
gray=[130 130 130]/255;
lightgray=[170 170 170]/255;
silver = [30 30 30]/255;
blue=[49,130,189]/255;
pink=[254 127 156]/255;
mediumblue=[128,207,231]/255;
lightblue=[158,202,225]/255;
red=[222,45,38]/255;
mediumred=[237,95,76]/255;
%lightred=[252,146,114]/255;
lightred=[251,177,168]/255;
ms=6;

switch(styID)
    case 'default'
        colours                 = {'blue','green','red','orange','aqua','magenta','yellow','black'};
        opt.general.linestyle   = {'-','-','-','-','-','-','-','-',...
                                   '--','--','--','--','--','--','--','--'};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.legend.leglocation  = 'eastoutside';
    case 'gray'
        colours                 = {black,gray,lightgray,silver,red};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v','s','^'};
        opt.general.linestyle   = {'-','--','-.','--','-.'};    
        opt.general.errorbars   = 'shade';
        opt.general.markersize  = 4;
    case 'gray_line'
        colours                 = {black,gray,lightgray,silver,red};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v','s','^'};
        opt.general.linestyle   = {'-','--','-.','--','-.'};    
        opt.general.markersize  = 4;
    case 'Two'
        % set different shades for colours - colour-blind friendly
        colours                 = {red,blue};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v'};
        opt.general.markerfacecolor = colours;
        opt.general.linestyle   = {'-','--'};
        opt.general.markersize  = 2;
    case 'TwoShade'
        colours                 = {red,blue};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v'};
        opt.general.linestyle   = {'-','--'};
        opt.general.errorbars   = 'shade';
    case 'Alter'
        colours                 = {red,blue,mediumred,mediumblue,lightred,lightblue};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','--','--'};
        opt.general.markersize  = ms;
    case 'AlterShade'
        colours                 = {red,blue,mediumred,mediumblue,lightred,lightblue};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','--','--'};
        opt.general.markersize  = 4;
        opt.general.errorbars   = 'shade';
    case 'AlterShade3'
        colours                 = {red,blue,mediumred,mediumblue,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','--','--','-.'};
        opt.general.markersize  = 4;
        opt.general.errorbars   = 'shade';       
    case 'Alter3'
        colours                 = {red,blue,mediumred,mediumblue,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','--','--','-.'};
        opt.general.markersize  = 4;
    case 'Alter4'
        colours                 = {red,mediumred,blue,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','--','--','-.'};
        opt.general.markersize  = 4;
        opt.general.errorbars   = 'shade';  
    case 'Alter3x3'
        colours                 = {red,red,red,blue,blue,blue,gray,gray,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v','s','o','v','s','o','v','s'};
        opt.general.linestyle   = {'-','--','-.'};
        opt.general.markersize  = 4;
        opt.general.errorbars   = 'shade';  
    case 'ThreeShade'
        colours                 = {red,blue,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','v','s'};
        opt.general.linestyle   = {'-','-'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 4;
    case 'FourShade'
        colours                 = {red,mediumred,blue,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','^','v','s'};
        opt.general.linestyle   = {'-','-'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 4;
    case 'FourShade_cool'
        colours                 = {[73 79 162]/255,lightblue,[126 202 170]/255,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','--','-','-'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 4;
    case 'FiveShade_cool'
        colours                 = {[73 79 162]/255,lightblue,[126 202 170]/255,gray,mediumred};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','--','-','-'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 4;
    case 'FourColor'
        colours = {black,gray,blue,red};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','-','-','--','--','--','--'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 4;
    case 'ThesisFour'
        colours = {gray,pink,blue,red};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','-','-','--','--','--','--'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 4;
    case 'FourColor_twoTypes'
        colours                  = {[73 79 162]/255,lightblue,mediumred,[126 202 170]/255};
        opt.general.markersize   = 4;
    case 'FourColor_wBlack'
        colours                  = {black,[73 79 162]/255,lightblue,[126 202 170]/255};
        opt.general.markersize   = 4;
    case 'FiveColours'
        colours                 = {[73 79 162]/255,lightblue,[126 202 170]/255,mediumred,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','-','-'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 3;
    case 'FourColours'
        colours                 = {lightblue,[126 202 170]/255,mediumred,gray};
        canvas                  = 'blackonwhite';
        opt.save.journal        = 'brain';
        opt.general.markertype  = {'o','s'};
        opt.general.linestyle   = {'-','-','-','-'};
        opt.general.errorbars   = 'shade';
        opt.general.markersize   = 3;
end;
