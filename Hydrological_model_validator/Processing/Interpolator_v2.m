function Interpolator_v2(varname, data_level, input_dir, output_dir, mask_file)

%% 
fprintf("Running the MatLab interpolator script...")
fprintf("As a reminder the chosen variable is:")
disp(['Variable name: ', varname]);
fprintf("And the chosen data level is:")
% Example usage of data_level and varname:
disp(['Using satellite data level: ', data_level]);
%% 

%-----IMPORT THE DATA-----
fprintf("Attemping to load the data from the Python output...");
fprintf("\n!!! Please verify that the paths match the ones from the Python scripts !!!");
disp(['\nThe data is in the fodler: ', input_dir])
fprintf('\n%s\n', repmat('-', 1, 45));
fprintf("Loading the SatData_clean.mat file...");
data = load(fullfile(input_dir, 'SatData_clean.mat'));
fprintf("\nThis dataset contains the following variables:\n");
% List all the field names (variable names) in the loaded structure
disp(fieldnames(data));
fprintf("\nSatData_clean data succesfully loaded");
fprintf('\n%s\n', repmat('-', 1, 45));

fprintf("Loading the ModData_complete.mat file...");
ModData_complete_temp = load(fullfile(input_dir, 'ModData_complete.mat'));
ModData_complete = double(ModData_complete_temp.ModData_complete);
fprintf("\nModData_complete data succesfully loaded");
fprintf('\n%s\n', repmat('-', 1, 45));

%-----EXTRACTING THE SINGULAR FIELDS FROM DATA-----
fprintf("Extracting the singular variables from data... \n");
Truedays = size(ModData_complete, 1);
fprintf("Trueday has been extracted! \n");
Sat_lon = double(data.Sat_lon);
fprintf("Slon has been extracted! \n");
Sat_lat = double(data.Sat_lat);
fprintf("Slat has been extracted! \n");
SatData_complete = double(data.SatData_complete);
fprintf("SatData_complete has been extracted! \n");

%-----FIND DAYS WITH NO SATELLITE OBSERVATIONS-----
field_nan = 0;
fprintf("Computing the number of days with missing satellite observations...\n")
for t = 1:Truedays
    string = strcat("Checking day ", num2str(t));
    disp(string)
    temporary1(t) = 1;
    temporary2 = nansum(nansum(SatData_complete(t, :, :)));
    if temporary2 == 0
        field_nan = field_nan + 1;
        temporary1 = nan;
    end
end
string = strcat("The satnan index has been obtained!");
disp(string);
satnan = find(isnan(temporary1));
clear temporary1 temporary2
%% 

%-----MODEL LAND SEA MASK-----
MASK = fullfile(mask_file, 'mesh_mask.nc');
mask3d = nc_varget(MASK, 'tmask');
mask3d = squeeze(mask3d);

%-----ELIMINATE DEGENERATE DIMENSION-----
Mmask = squeeze(mask3d(1, :, :));

%-----FIND LAND GRIDPOINTS INDEXES FROM MODEL MASK-----
Mfsm = find(Mmask == 0);
Mfsm_3d = find(mask3d == 0);

%-----GET MODEL LAT & LON-----
Mask_lat = nc_varget(MASK, 'nav_lat');
Mask_lon = nc_varget(MASK, 'nav_lon');

%-----INITIALIZE AVERAGE TIME SERIES-----
Model_TS_avg(1:Truedays) = nan;
Satellite_TS_avg(1:Truedays) = nan;
%% 

if strcmp(data_level, "l4")
    %-----IF "L4" SATELLITE DATA IS INTERPOLATED ON MODEL GRID-----
    tmp = squeeze(SatData_complete(1, :, :));
    satmask = abs(isnan(tmp) - 1);
    Sat_lat = Sat_lat';
    clear tmp;

    %-----ITERATE OVER DAYS-----
    fprintf("Proceeding with the interpolation of the L4 (optimally interpolated) data");
    disp(string);

    for d = 1:Truedays
        noflood = squeeze(SatData_complete(d, :, :));

        %-----EXPAND DATA OVER LAND-----
        flooded = Flood(noflood, 5);

        %-----INTERPOLATE INTO MODEL GRID-----
        Sat_mask_temp = interp2(Sat_lon, Sat_lat, flooded, Mask_lon, Mask_lat);

        %-----MASK FIELDS-----
        Sat_mask_temp(Mfsm) = nan;
        SatData_interp(d, :, :) = Sat_mask_temp(:, :);

        string = strcat("Interpolating day ", num2str(d));
        disp(string);
    end

    string = "Interpolation terminated";
    disp(string);

    %-----SAVE AS NetCDF FILES-----
    fprintf("Setting up the Output Folder...\n");
    interPath = output_dir;
    fprintf("Proceding to save the data...\n")

    fprintf("Saving the model level 4 data...\n")
    % Check if the file exists, if not create it
    ncfile = [interPath, '\', 'ModData_', varname, '_interp_l4.nc'];

    % Create the NetCDF file and the variable 'ModData_complete' if they don't exist
    if exist(ncfile, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'ModData_complete'
    nccreate(ncfile, 'ModData_complete', 'Dimensions', {'time', Truedays, 'lat', size(ModData_complete, 2), 'lon', size(ModData_complete, 3)});

    % Write the data to the 'ModData_interp' variable in the NetCDF file
    ncwrite(ncfile, 'ModData_complete', ModData_complete);
    fprintf("ModData_complete_l4.nc file has been saved\n");

    fprintf("Saving the interpolated satellite level 4 data...\n")
    % Saving SatData_interp.nc file similarly
    ncfile2 = [interPath, '\', 'SatData_', varname, '_interp_l4.nc'];

    if exist(ncfile2, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile2); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'SatData_interp'
    nccreate(ncfile2, 'SatData_interp', 'Dimensions', {'time', Truedays, 'lat', size(SatData_interp, 2), 'lon', size(SatData_interp, 3)});

    % Write the SatData_interp data
    ncwrite(ncfile2, 'SatData_interp', SatData_interp);
    fprintf("SatData_interp_l4.nc file has been created\n")

    fprintf("Level 4 interpolated data has been saved!\n");

elseif strcmp(data_level, "l3")
    Sat_lat = Sat_lat';

    %-----IF "L3" MODEL DATA INTERPOLATED ON SATELLITE GRID-----
    MinMlat = min(min(Mask_lat));
    exSgrid = find(Sat_lat <= MinMlat);

    string = "Proceding with a bilinear interpolation of the L3s data";
    disp(string);

    for d = 1:Truedays
        string = strcat("Interpolating day ", num2str(d));
        disp(string);

        %-----GENERATE A LAND SEA MASK FOR SAT FIELDS-----
        Sat_mask_temp = squeeze(SatData_complete(d, :, :));
        Sat_mask_temp(exSgrid) = nan;
        switch varname
            case 'chl'
                outlierconc = 15;
            case 'sst'
                outlierconc = 35;
        end
        outliers = find(Sat_mask_temp >= outlierconc);
        Sat_mask_temp(outliers) = nan;
        SatData_complete(d, :, :) = Sat_mask_temp(:,:);
        satmask = abs(isnan(Sat_mask_temp) - 1);
        nobs = nansum(nansum(satmask));
        if nobs <= 500
            satmask(:,:) = 0;
        end
        satmasknan = find(satmask == 0);
        SatData_complete(d, :, :) = Sat_mask_temp(:,:);
        SatData_complete(satmasknan) = nan;
        clear Sat_mask_temp;

        %-----ELIMINATE DATA WITH NO SATELLITE OBSERVATIONS-----
        ModData_complete(satnan, :, :) = nan;
        noflood = squeeze(ModData_complete(d, :, :));

        %-----EXPAND DATA OVER LAND-----
        flooded = Flood(noflood, 5);

        %-----INTERPOLATE INTO SATELLITE GRID-----
        Mtmp = interp2(Mask_lon, Mask_lat, flooded, Sat_lon, Sat_lat);

        %-----MASK FIELDS-----
        Mtmp(outliers) = nan;
        mgp = find(satmask == 0);
        Mtmp(mgp) = nan;
        outliers = find(Mtmp >= outlierconc);
        Mtmp(outliers) = nan;
        clear mgp;

        %-----STORE INTERPOLATED DATA-----
        ModData_interp(d, :, :) = Mtmp(:,:);

    end
%%
    %-----SAVE AS NetCDF FILES-----
    fprintf("Setting to the Output Folder...\n");
    interPath = output_dir;
    fprintf("Proceding to save the data...\n")

    fprintf("Saving the interpolated level 3 data...\n")
    % Check if the file exists, if not create it
    ncfile = [interPath, '\', 'ModData_', varname, '_interp_l3.nc'];

    % Create the NetCDF file and the variable 'ModData_interp' if they don't exist
    if exist(ncfile, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'ModData_interp'
    nccreate(ncfile, 'ModData_interp', 'Dimensions', {'time', Truedays, 'lat', size(ModData_interp, 2), 'lon', size(ModData_interp, 3)});

    % Write the data to the 'ModData_interp' variable in the NetCDF file
    ncwrite(ncfile, 'ModData_interp', ModData_interp);
    fprintf("ModData_interp_l3.nc file has been saved\n");

    fprintf("Saving the satellite level 3 data...\n")
    % Saving SatData_complete.nc file similarly
    ncfile2 = [interPath, '\', 'SatData_', varname, '_interp_l3.nc'];

    if exist(ncfile2, 'file') == 2
        fprintf("File already exists, overwriting...\n");
        delete(ncfile2); % Optional: delete the existing file if overwriting is desired
    end

    % Create the NetCDF file and variable for 'SatData_complete'
    nccreate(ncfile2, 'SatData_complete', 'Dimensions', {'time', Truedays, 'lat', size(SatData_complete, 2), 'lon', size(SatData_complete, 3)});

    % Write the SatData_complete data
    ncwrite(ncfile2, 'SatData_complete', SatData_complete);
    fprintf("SatData_interp_l3.nc file has been created\n")

    fprintf("Level 3 interpolated data has been saved!\n");

else
    string = "Problem with data_level!!!!";
    disp(string);
    return;
end

string = "The data has been succesfully interpolated!";
disp(string);
fprintf('\n%s\n', repmat('-', 1, 45));
%% 

% ----- COMPUTING BASIN AVERAGES -----

fprintf("Creating the Basin Average timeseries for the selected variable and data level...")

DafterD=0; % Initializing a counter to keep track of the days in the dataset

% ----- ALLOCATE THE ARRAY -----

BAmod(1:Truedays)=0;
BAsat(1:Truedays)=0;

for d=1:Truedays
    string = strcat("Averaging day ", num2str(d));
    disp(string);

    DafterD=DafterD+1;

    % ----- INITIALIZING AND EXTENDING THE ARRAYS -----

    switch data_level
        case 'l4'
            ModData = squeeze(ModData_complete(DafterD, :, :));
        case 'l3'
            ModData = squeeze(ModData_interp(DafterD, :, :));
        otherwise
            fprintf("Invalid data_level\n");
    end

    % ----- APPLY THE MASK AND LOOK FOR NaN -----

    switch data_level
        case 'l4'
            SatData = squeeze(SatData_interp(DafterD, :, :));
        case 'l3'
            SatData = squeeze(SatData_complete(DafterD, :, :));
        otherwise
            fprintf("Data never to be seen");
    end

    SatData_fsm=find(isnan(SatData));

    switch data_level
        case 'l4'
            ModData(Mfsm) = nan;
            ModData(SatData_fsm) = nan;
        case 'l3'
            ModData(SatData_fsm) = nan;       
        otherwise
            disp("NaN'ing: NEVER TO BE SEEN");
    end

    % ----- COMPUTE THE BASIN AVERAGES

    BAmod(DafterD)=nanmean(nanmean(ModData));
    BAsat(DafterD)=nanmean(nanmean(SatData));

    %        -----ELIMATE "INCONSISTENCIES"-----

    if BAmod(DafterD) == 0
        BAmod(DafterD)=nan;
        BAsat(DafterD)=nan;
    end

end

nomod=find(isnan(BAmod));
BAsat(nomod)=nan;

switch data_level
    case 'l4'
        BAmod_L4 = BAmod;
        BAsat_L4 = BAsat;
    case 'l3'
        BAmod_L3 = BAmod;
        BAsat_L3 = BAsat;
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
interpath = output_dir;
fprintf("Proceding to save the data...\n")

switch data_level
    case 'l4'
        fprintf("Saving the Basin Average Model level 4 data...\n");
        ncfile = [interpath, '\', 'BA_', varname, '_mod_L4.nc'];
        if exist(ncfile, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile);
        end

        nccreate(ncfile, 'BAmod_L4', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile, 'BAmod_L4', BAmod_L4);
        fprintf("The Basin Average Model level 4 data has been saved!\n");

        fprintf("Saving the Basin Average Satellite level 4 data...\n");
        ncfile2 = [interpath, '\', 'BA_', varname, '_sat_L4.nc'];
        if exist(ncfile2, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile2);
        end

        nccreate(ncfile2, 'BAsat_L4', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile2, 'BAsat_L4', BAsat_L4);
        fprintf("The Basin Average Satellite level 4 data has been saved!\n");

    case 'l3'
        fprintf("Saving the Basin Average Model level 3 data...\n");
        ncfile = [interpath, '\', 'BA_' varname, '_mod_L3.nc'];
        if exist(ncfile, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile);
        end

        nccreate(ncfile, 'BAmod_L3', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile, 'BAmod_L3', BAmod_L3);
        fprintf("The Model Basin Average level 3 data has been saved!\n");

        fprintf("Saving the Basin Average Satellite level 3 data...\n");
        ncfile2 = [interpath, '\', 'BA_', varname, '_sat_L3.nc'];
        if exist(ncfile2, 'file') == 2
            fprintf("File already exists, overwriting...\n");
            delete(ncfile2);
        end

        nccreate(ncfile2, 'BAsat_L3', 'Dimensions', {'time', Truedays}, 'Datatype', 'double');
        ncwrite(ncfile2, 'BAsat_L3', BAsat_L3);
        fprintf("The Basin Average Satellite  level 3 data has been saved!\n");

    otherwise
        fprintf("Error in the chl level selected!\n");
end