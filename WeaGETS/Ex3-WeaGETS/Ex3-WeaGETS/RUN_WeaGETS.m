% Original code: https://ch.mathworks.com/matlabcentral/fileexchange/29136-stochastic-weather-generator-weagets
% ************************************************************************
% Weather Generator Ecole de Technologie Superieure (WeaGETS)
% ************************************************************************
%
% WeaGETS is a Matlab-based versatile stochastic daily weather generator
% for producing daily precipitation, maximum and minimum temperatures
% (Tmax and Tmin) series of unlimited length, thus permitting impact
% studies of rare occurrences of meteorological variables. Furthermore, it
% can be used in climate change studies as a downscaling tool by perturbing
% their parameters to account for expected changes in precipitation and
% temperature. First, second and third-order Markov models are provided to
% generate precipitation occurrence, and four distributions (exponential,
% gamma, skewed normal and mixed exponential) are available to produce
% daily precipitation quantity. Precipitation generating parameters have
% options to be smoothed using Fourier harmonics.
% Two schemes (unconditional and conditional) are available to simulate
% Tmax and Tmin. Finally, a spectral correction approach is included to
% correct the well-known underestimation of monthly and inter-annual
% variability associated with weather generators.
%% Input data
% ****************************
% The input data consists of daily precipitation, Tmax and Tmin. The model
% does not take into account bissextile years. Any significant precipitation
% occurring on a February 29th should be redistributed equally on February
% 28th and March 1st. The maximum and minimum temperatures of a February
% 29th can be simply removed. Missing data should be assigned a -999 value.
% The input file contains the following matrices and vectors:
% (1)	P: matrix with dimensions [nyears*365], where nyears is the number
%          of years, containing daily precipitation in mm.
% (2)	Tmax: matrix with dimensions [nyears *365], where nyears is the
%             number of years, containing maximum temperature in Celsius.
% (3)	Tmin: matrix with dimensions [nyears *365], where nyears is the
%             number of years, containing minimum temperature in Celsius.
% (4)	yearP: vector of length [nyears *1] containing the years covered
%              by the precipitation.
% (5)	yearT: vector of length [nyears *1] containing the years covered
%              by the Tmax and Tmin.
%% Output data
% ****************************
% The output also consists of daily precipitation, Tmax and Tmin values.
% It contains the following matrices:
% (1)	gP: matrix with dimensions [gnyears*365], where gnyears is the
%           number of years of generated precipitation in mm without
%           low-frequency variability correction.
% (2)	gTmax: matrix with dimensions [gnyears *365], where gnyears is the
%              number of years of generated Tmax in Celsius without
%              low-frequency variability correction.
% (3)	gTmin: matrix with dimensions [gnyears *365], where gnyears is the
%              number of years of generated Tmin in Celsius without
%              low-frequency variability correction.
%% Running the program
% ****************************
% There are many subprograms in the WeaGETS package, but the user only
% needs to run the main program RUN_WeaGETS.m. All of the options will
% then be offered in the form of questions, presented as follows:
% (1)	Basic input
% a.	Enter an input file name (string):
% A name for the observed data shall be entered within single quotes, for
% instance, ??filename?? for the supplied file.
% b.	Enter an output file name (string):
% A name for the generated data shall be entered within single quotes, for
% example ??filename_generated??.
% c.	Enter a daily precipitation threshold:
% Precipitation threshold is the amount of precipitation used to determine
% whether a given day is wet or not (0.1mm is the most commonly used value).
% d.	Enter the number of years to generate:
% The number of years of the generated time series of precipitation and
% temperatures is entered here.
% (2)	Precipitation and temperature generation
% a.	Select an order of Markov Chain to generate precipitation
%       occurrence, 1: First-order; 2: Second-order; 3: Third-order.
% b.	Select a distribution to generate wet day precipitation amount:
%       1: Exponential, 2: Gamma, 3: Skewed normal or 4: Mixed exponential.
%% WeaGETS went through several iterations.  Prof. Robert Leconte (now at
% Sherbrooke University) wrote the original Matlab code based on WGEN
% (Richardson and Wright, 1984).  Prof. Francois Brissette then modified
% the code and streamlined it close to its current form.  Master student
% Annie Caron tested several aspects of the code and added higher order
% Markov Chains for precipitation occurrence (Caron et al., 2008).
% Finally, PhD student Jie Chen provided several additional options
% including the correction scheme for the well know problem of the
% underestimation of inter annual variability (Chen et al., 2010), and the
% CLIGEN temperature scheme (Chen et al., 2010).
%% References:
% (1) Caron, A., Leconte, R., Brissette, F.P, 2008. Calibration and
% validation of a stochastic weather generator for climate change studies.
% Canadian Water Resources Journal. 33(3): 233-256.
% (2) Chen J., Brissette, P.F., Leconte, R., 2011. Assessment and
% improvement of stochastic weather generators in simulating maximum and
% minimum temperatures.Transactions of the ASABE, 54 (5), 1627-1637.
% (3) Chen J., Zhang, X.C., Liu, W.Z., Li, Z., 2008. Assessment and
% Improvement of CLIGEN Non-Precipitation Parameters for the Loess Plateau
% of China.Transactions of the ASABE 51(3), 901-913.
% (4) Chen, J., Brissette, P.F., Leconte, R., 2010. A daily stochastic
% weather generator for preserving low-frequency of climate variability.
% Journal of Hydrology 388, 480-490.
% (5) Chen, J., Brissette, P.F., Leconte, R., Caron, A. 2012. A versatile
% weather generator for daily precipitation and temperature. Transactions
% of the ASABE, 55(3), 895-906.
% (6) Nicks, A.D., Lane, L.J., Gander, G.A., 1995. Weather generator, Ch. 2.
% In USDA?Water Erosion Prediction Project: Hillslope Profile and Watershed
% Model Documentation, eds. D. C. Flanagan, and M. A. Nearing. NSERL Report
% No. 10. West Lafayette, Ind.: USDA-ARS-NSERL.
% (7) Richardson, C.W., 1981. Stochastic simulation of daily precipitation,
% temperature, and solar radiation. Water Resources Research 17, 182-190.
% (8) Richardson, C.W., Wright, D.A., 1984. WGEN: A model for generating
% daily weather variables. U.S. Depart. Agr, Agricultural Research Service.
% Publ. ARS-8.
%% Code
clear
%% Basic input
% basic inputs include names of input and output files, daily precipitation
% threshold and a number of years for generated data
filenamein=uigetfile('*.mat','Select an input filename');
filenameout=uiputfile('*.mat','Enter an output filename');
PrecipThreshold=inputdlg({'Enter a daily precipitation threshold:'},'Input:',[1 35],{'0.1'});
PrecipThreshold=str2num(PrecipThreshold{1});
GeneratedYears=inputdlg({'Enter the number of years to generate:'},'Input:',[1 35],{'30'});
GeneratedYears=str2num(GeneratedYears{1});
%% Precipitation and temperatures generation
MarkovChainOrder = questdlg('Select an order of Markov Chain to generate precipitation occurrence','Select','First-order','Second-order','Third-order','First-order');
switch MarkovChainOrder
    case 'First-order'
        MarkovChainOrder = 1;
    case 'Second-order'
        MarkovChainOrder = 2;
    case 'Third-order'
        MarkovChainOrder = 3;
end
idistr = listdlg('PromptString','Select a distribution to generate wet day precipitation amount','SelectionMode','single','ListString',{'Exponential','Gamma','Skewed normal','Mixed exponential'});
% the parameters are analysed at 2 weeks scale for each option of
% Markov Chain order
if MarkovChainOrder==1  % first order Markov Chain
    [idistr,ap00,ap10,par,A,B,aC0,aC1,aC2,aD1,aD2,sC0,sC1,sC2,sD1,...
        sD2,PrecipThreshold,MarkovChainOrder]=analyzer_order1_norad_no(filenamein,PrecipThreshold,idistr);
elseif MarkovChainOrder==2  % second order Markov Chain
    [idistr,ap000,ap010,ap100,ap110,par,A,B,aC0,aC1,aC2,...
        aD1,aD2,sC0,sC1,sC2,sD1,sD2,PrecipThreshold,MarkovChainOrder]=analyzer_order2_norad_no...
        (filenamein,PrecipThreshold,idistr);
elseif MarkovChainOrder==3  % third order Markov Chain
    [idistr,ap0000,ap0010,ap0100,ap0110,ap1000,ap1010,ap1100,...
        ap1110,par,A,B,aC0,aC1,aC2,aD1,aD2,sC0,sC1,sC2,sD1,sD2,PrecipThreshold,...
        MarkovChainOrder]=analyzer_order3_norad_no(filenamein,PrecipThreshold,idistr);
end
TempScheme=1;
% run weather generator to produce sythesize data
if MarkovChainOrder==1  % first order Markov Chain
    generator_order1_norad_no(filenameout,GeneratedYears,TempScheme,idistr,...
        ap00,ap10,par,A,B,aC0,aC1,aC2,aD1,aD2,sC0,sC1,sC2,sD1,...
        sD2,PrecipThreshold,MarkovChainOrder);
elseif MarkovChainOrder==2  % second order Markov Chain
    generator_order2_norad_no(filenameout,GeneratedYears,TempScheme,...
        idistr,ap000,ap010,ap100,ap110,par,A,B,aC0,aC1,aC2,...
        aD1,aD2,sC0,sC1,sC2,sD1,sD2,PrecipThreshold,MarkovChainOrder);
elseif MarkovChainOrder==3  % third order Markov Chain
    generator_order3_norad_no(filenameout,GeneratedYears,TempScheme,idistr,...
        ap0000,ap0010,ap0100,ap0110,ap1000,ap1010,ap1100,ap1110,par,A,B,...
        aC0,aC1,aC2,aD1,aD2,sC0,sC1,sC2,sD1,sD2,PrecipThreshold,MarkovChainOrder);
end