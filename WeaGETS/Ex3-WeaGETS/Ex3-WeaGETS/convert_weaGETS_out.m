function [TS] = convert_weaGETS_out(gP, gTmax, gTmin, d)
%Converts the WeaGETS output into a cell array of time series, of the
%desired length.
%   Input arguments are:
%           -gP    - Daily precipitation time series
%           -gTmax - Daily maximum temperature
%           -gTmin - Daily minimum temperature
%           -d     - Time series duration for extraction [yr]
%   The output consists of n cells containing the matrices of time series
%   of length d (set as an input parameter). Each cell contains d*365 X 4
%   data points, arranged in columns: Julian Day, Precipitation, max.
%   Temperature, min. Temperature.

% Check if input variables have the same length.
if ~(isequal(size(gP), size(gTmax)) && isequal(size(gP), size(gTmin)))
    error('Your input time series must be of same length.')
end
% Check if input variables length is dividible by d.
if mod(size(gP,1),d)~=0
    error('Your input time series length must be a multiple of d. Choose a different d value or re-run your WeaGETS generator with an appropriate series length.')
end
%Create the TS cell array
TS = cell(size(gP,1)/d,1);
n = length(TS);
l = size(gP,1);

    j = d;
    P = arrayfun(@(j) reshape(gP(j:j+d-1,:).',1,[])', 1:d:l-d+1,'UniformOutput',false); 
    Tmax = arrayfun(@(j) reshape(gTmax(j:j+d-1,:).',1,[])', 1:d:l-d+1,'UniformOutput',false); 
    Tmin = arrayfun(@(j) reshape(gTmin(j:j+d-1,:).',1,[])', 1:d:l-d+1,'UniformOutput',false); 

for i = 1:n
    TS{i} = [repmat(1:365,1,d)', P{i}, Tmax{i}, Tmin{i}];
end

end