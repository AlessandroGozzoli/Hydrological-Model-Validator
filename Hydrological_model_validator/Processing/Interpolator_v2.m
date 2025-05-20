%-----WORKING DIRECTORY-----
%
WDIR = "C:/Hydrological_model_validator";
%% 

%-----SATELLITE DATA LEVEL (NEEDED FOR CHLOROPHYLL DATA)-----
% N.B: chldlev='l3' data level 3
%      chldlev='l4' data level 4
chldlev = "l4";
%% 

%-----IMPORT THE DATA-----
fprintf("Attemping to load the data from the Python output...");
fprintf("\n!!! Please verify that the paths match the ones from the Python scripts !!!");
fprintf('\n%s\n', repmat('-', 1, 45));
fprintf("Loading the chl_clean.mat file...");
data = load('C:/Hydrological_model_validator/Data/INTERPOLATOR_INPUT/chl_clean.mat');
fprintf("\nThis dataset contains the following variables:\n");
% List all the field names (variable names) in the loaded structure
disp(fieldnames(data));
fprintf("\nchl_clean data succesfully loaded");
fprintf('\n%s\n', repmat('-', 1, 45));

fprintf("Loading the Mchl_complete.mat file...");
Mchl_complete_temp = load('C:/Hydrological_model_validator/Data/INTERPOLATOR_INPUT/Mchl_complete.mat');
Mchl_complete = double(Mchl_complete_temp.Mchl_complete);
fprintf("\nMchl_complete data succesfully loaded");
fprintf('\n%s\n', repmat('-', 1, 45));

%-----EXTRACTING THE SINGULAR FIELDS FROM DATA-----
fprintf("Extracting the singular variables from data... \n");
Truedays = 3653;
fprintf("Trueday has been extracted! \n");
Slon = double(data.Slon);
fprintf("Slon has been extracted! \n");
Slat = double(data.Slat);
fprintf("Slat has been extracted! \n");
Schl_complete = double(data.Schl_complete);
fprintf("Schl_complete has been extracted! \n");

%-----FIND DAYS WITH NO SATELLITE OBSERVATIONS-----
cnan = 0;
for t = 1:Truedays
    tempo1(t) = 1;
    tempo2 = nansum(nansum(Schl_complete(t, :, :)));
    if tempo2 == 0
        cnan = cnan + 1;
        tempo1 = nan;
    end
end
string = strcat("The satnan index has been obtained!");
disp(string);
satnan = find(isnan(tempo1));
clear tempo1 tempo2
%% 

%-----MODEL LAND SEA MASK-----
MASK = 'C:/Tesi magistrale/Dati/mesh_mask.nc';
mask3d = nc_varget(MASK, 'tmask');
mask3d = squeeze(mask3d);

%-----ELIMINATE DEGENERATE DIMENSION-----
Mmask = squeeze(mask3d(1, :, :));

%-----FIND LAND GRIDPOINTS INDEXES FROM MODEL MASK-----
Mfsm = find(Mmask == 0);
Mfsm_3d = find(mask3d == 0);

%-----GET MODEL LAT & LON-----
Mlat = nc_varget(MASK, 'nav_lat');
Mlon = nc_varget(MASK, 'nav_lon');

%-----INITIALIZE AVERAGE TIME SERIES-----
M_TS_ave(1:Truedays) = nan;
S_TS_ave(1:Truedays) = nan;
%% 

if strcmp(chldlev, "l4")
    %-----IF "L4" SATELLITE DATA IS INTERPOLATED ON MODEL GRID-----
    tmp = squeeze(Schl_complete(1, :, :));
    satmask = abs(isnan(tmp) - 1);
    Slat = Slat';
    clear tmp;

    %-----ITERATE OVER DAYS-----
    string = "Start interpolation loop for the level 4 data";
    disp(string);

    for d = 1:Truedays
        noflood = squeeze(Schl_complete(d, :, :));

        %-----EXPAND DATA OVER LAND-----
        flooded = Flood(noflood, 5);

        %-----INTERPOLATE INTO MODEL GRID-----
        Stmp = interp2(Slon, Slat, flooded, Mlon, Mlat);

        %-----MASK FIELDS-----
        Smtp(Mfsm) = nan;
        Schl_interp(d, :, :) = Stmp(:, :);

        string = strcat("Interpolating day ", num2str(d));
        disp(string);
    end

    string = "Interpolation terminated";
    disp(string);

    %-----SAVE AS NetCDF FILES-----
    fprintf("Setting to the Output Folder...\n");
    interPath = 'C:/Hydrological_model_validator/DatA/OUTPUT/INTERPOLATOR/';
    fprintf("Proceding to save the data...\n")

    fprintf("Saving the Mchl level 4 data...\n")
    % Check if the file exists, if not create it
    ncfile = [interPath 'Mchl_interp_l4.nc'];

    % Create the NetCDF file and the variable 'Mchl_interp' if they don't exist
    if exist(ncfile, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'Mchl_interp'
    nccreate(ncfile, 'Mchl_complete', 'Dimensions', {'time', Truedays, 'lat', size(Mchl_complete, 2), 'lon', size(Mchl_complete, 3)});

    % Write the data to the 'Mchl_interp' variable in the NetCDF file
    ncwrite(ncfile, 'Mchl_complete', Mchl_complete);
    fprintf("Mchl_interp_l4.nc file has been saved\n");

    fprintf("Saving the Schl level 4 data...\n")
    % Saving Schl_interp_l3.nc file similarly
    ncfile2 = [interPath 'Schl_interp_l4.nc'];

    if exist(ncfile2, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile2); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'Schl_complete'
    nccreate(ncfile2, 'Schl_interp', 'Dimensions', {'time', Truedays, 'lat', size(Schl_interp, 2), 'lon', size(Schl_interp, 3)});

    % Write the Schl_complete data
    ncwrite(ncfile2, 'Schl_interp', Schl_interp);
    fprintf("Schl_interp_l4.nc file has been created\n")

    fprintf("Level 4 interpolated data has been saved!\n");

elseif strcmp(chldlev, "l3")
    Slat = Slat';

    %-----IF "L3" MODEL DATA INTERPOLATED ON SATELLITE GRID-----
    MinMlat = min(min(Mlat));
    exSgrid = find(Slat <= MinMlat);

    string = "Interpolating the Model data onto the satellite grid";
    disp(string);

    for d = 1:Truedays
        string = strcat("Interpolating day ", num2str(d));
        disp(string);

        %-----GENERATE A LAND SEA MASK FOR SAT FIELDS-----
        Stmp = squeeze(Schl_complete(d, :, :));
        Stmp(exSgrid) = nan;
        outlierconc = 15;
        outliers = find(Stmp >= outlierconc);
        Stmp(outliers) = nan;
        Schl_complete(d, :, :) = Stmp(:,:);
        satmask = abs(isnan(Stmp) - 1);
        nobs = nansum(nansum(satmask));
        if nobs <= 500
            satmask(:,:) = 0;
        end
        satmasknan = find(satmask == 0);
        Schl_complete(d, :, :) = Stmp(:,:);
        Schl_complete(satmasknan) = nan;
        clear Stmp;

        %-----ELIMINATE DATA WITH NO SATELLITE OBSERVATIONS-----
        Mchl_complete(satnan, :, :) = nan;
        noflood = squeeze(Mchl_complete(d, :, :));

        %-----EXPAND DATA OVER LAND-----
        flooded = Flood(noflood, 5);

        %-----INTERPOLATE INTO SATELLITE GRID-----
        Mtmp = interp2(Mlon, Mlat, flooded, Slon, Slat);

        %-----MASK FIELDS-----
        Mtmp(outliers) = nan;
        mgp = find(satmask == 0);
        Mtmp(mgp) = nan;
        outliers = find(Mtmp >= outlierconc);
        Mtmp(outliers) = nan;
        clear mgp;

        %-----STORE INTERPOLATED DATA-----
        Mchl_interp(d, :, :) = Mtmp(:,:);

    end
%%
    %-----SAVE AS NetCDF FILES-----
    fprintf("Setting to the Output Folder...\n");
    interPath = 'C:/Hydrological_model_validator/Data/OUTPUT/INTERPOLATOR/';
    fprintf("Proceding to save the data...\n")

    fprintf("Saving the Mchl level 3 data...\n")
    % Check if the file exists, if not create it
    ncfile = [interPath 'Mchl_interp_l3.nc'];

    % Create the NetCDF file and the variable 'Mchl_interp' if they don't exist
    if exist(ncfile, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'Mchl_interp'
    nccreate(ncfile, 'Mchl_interp', 'Dimensions', {'time', Truedays, 'lat', size(Mchl_interp, 2), 'lon', size(Mchl_interp, 3)});

    % Write the data to the 'Mchl_interp' variable in the NetCDF file
    ncwrite(ncfile, 'Mchl_interp', Mchl_interp);
    fprintf("Mchl_interp_l3.nc file has been saved\n");

    fprintf("Saving the Schl level 3 data...\n")
    % Saving Schl_interp_l3.nc file similarly
    ncfile2 = [interPath 'Schl_interp_l3.nc'];

    if exist(ncfile2, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile2); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'Schl_complete'
    nccreate(ncfile2, 'Schl_complete', 'Dimensions', {'time', Truedays, 'lat', size(Schl_complete, 2), 'lon', size(Schl_complete, 3)});

    % Write the Schl_complete data
    ncwrite(ncfile2, 'Schl_complete', Schl_complete);
    fprintf("Schl_interp_l3.nc file has been created\n")

    fprintf("Level 3 interpolated data has been saved!\n");

else
    string = "Problem with chldlev!!!!";
    disp(string);
    return;
end

string = "CHL data succesfully interpolated!";
disp(string);
fprintf('\n%s\n', repmat('-', 1, 45));
%% 

% ----- COMPUTING BASIN AVERAGES -----

fprintf("Creating the Basin Average timeseries for the selected data level...")

DafterD=0; % Initializing a counter to keep track of the days in the dataset

% ----- ALLOCATE THE ARRAY -----

BACHLmod(1:Truedays)=0;
BACHLsat(1:Truedays)=0;

for d=1:Truedays
    string = strcat("Averaging day ", num2str(d));
    disp(string);

    DafterD=DafterD+1;

    % ----- INITIALIZING AND EXTENDING THE ARRAYS -----

    switch chldlev
        case 'l4'
            Mchl = squeeze(Mchl_complete(DafterD, :, :));
        case 'l3'
            Mchl = squeeze(Mchl_interp(DafterD, :, :));
        otherwise
            fprintf("Invalid chldlev\n");
    end

    % ----- APPLY THE MASK AND LOOK FOR NaN -----

    switch chldlev
        case 'l4'
            Schl = squeeze(Schl_interp(DafterD, :, :));
        case 'l3'
            Schl = squeeze(Schl_complete(DafterD, :, :));
        otherwise
            fprintf("CHL: never to be seen");
    end

    Schlfsm=find(isnan(Schl));

    switch chldlev
        case 'l4'
            Mchl(Mfsm) = nan;
            Mchl(Schlfsm) = nan;
        case 'l3'
            Mchl(Schlfsm) = nan;       
        otherwise
            disp("NaN'ing: NEVER TO BE SEEN");
    end

    chlanom=Mchl-Schl; % Compute the anomalies

    % ----- COMPUTE THE BASIN AVERAGES

    BACHLmod(DafterD)=nanmean(nanmean(Mchl));
    BACHLsat(DafterD)=nanmean(nanmean(Schl));

    %        -----ELIMATE "INCONSISTENCIES"-----

    if BACHLmod(DafterD) == 0
        BACHLmod(DafterD)=nan;
        BACHLsat(DafterD)=nan;
    end

end

nomod=find(isnan(BACHLmod));
BACHLsat(nomod)=nan;

switch chldlev
    case 'l4'
        BACHLmod_L4 = BACHLmod;
        BACHLsat_L4 = BACHLsat;
    case 'l3'
        BACHLmod_L3 = BACHLmod;
        BACHLsat_L3 = BACHLsat;
    otherwise
        fprintf("Error in the chl level selected!");
end

if DafterD ~= Truedays
   disp ("There is a reading problem");
   fprintf('\n%s\n', repmat('-', 1, 45));
   return
else
   disp("Basin averages datasets computed, saving...");
   fprintf('%s\n', repmat('-', 1, 45));
end
%% 
fprintf("Setting to the Output Folder...\n");
interpath = 'C:/Hydrological_model_validator/Data/OUTPUT/INTERPOLATOR/';
fprintf("Proceding to save the data...\n")

switch chldlev
    case 'l4'
        fprintf("Saving the Basin Average Model CHL level 4 data...\n");
        ncfile = [interpath, 'BACHLmod_L4.nc'];
        if exist(ncfile, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile);
        end

        nccreate(ncfile, 'BACHLmod_L4', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile, 'BACHLmod_L4', BACHLmod_L4);
        fprintf("The Basin Average Model CHL level 4 data has been saved!\n");

        fprintf("Saving the Basin Average Satellite CHL level 4 data...\n");
        ncfile2 = [interpath, 'BACHLsat_L4.nc'];
        if exist(ncfile2, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile2);
        end

        nccreate(ncfile2, 'BACHLsat_L4', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile2, 'BACHLsat_L4', BACHLsat_L4);
        fprintf("The Basin Average Satellite CHL level 4 data has been saved!\n");

    case 'l3'
        fprintf("Saving the Basin Average Model CHL level 3 data...\n");
        ncfile = [interpath, 'BACHLmod_L3.nc'];
        if exist(ncfile, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile);
        end

        nccreate(ncfile, 'BACHLmod_L3', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile, 'BACHLmod_L3', BACHLmod_L3);
        fprintf("The Model Basin Average CHL level 3 data has been saved!\n");

        fprintf("Saving the Basin Average Satellite CHL level 3 data...\n");
        ncfile2 = [interpath, 'BACHLsat_L3.nc'];
        if exist(ncfile2, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile2);
        end

        nccreate(ncfile2, 'BACHLsat_L3', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile2, 'BACHLsat_L3', BACHLsat_L3);
        fprintf("The Basin Average Satellite CHL level 3 data has been saved!\n");

    otherwise
        fprintf("Error in the chl level selected!\n");
end

