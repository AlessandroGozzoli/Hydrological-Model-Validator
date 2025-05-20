%-----WORKING DIRECTORY-----
%
WDIR="C:/Tesi magistrale/Dati/OUTPUT/";
%
%-----SATELLITE DATA LEVEL (NEEDED FOR CHLOROPHYLL DATA)-----
%
% N.B: chldlev='l3' data level 3
%      chldlev='l4' data level 4
%
chldlev="l3";
%-----IMPORT THE DATA-----
%
fprintf("Attemping to load the data from the Python output...");
fprintf("\n!!! Please verify that the paths match the ones from the Python scripts !!!");
fprintf('\n%s\n', repmat('-', 1, 45));
%
fprintf("Loading the chl_clean.mat file...")
data = load('C:/Tesi magistrale/Dati/OUTPUT/SATELLITE_CLEAN/chl_clean.mat');
fprintf("\nThis dataset contains the following variables:\n");
% List all the field names (variable names) in the loaded structure
disp(fieldnames(data));
fprintf("\nchl_clean data succesfully loaded")
fprintf('\n%s\n', repmat('-', 1, 45));
%
fprintf("Loading the Mchl_complete.mat file...")
Mchl_complete_temp = load('C:/Tesi magistrale/Dati/OUTPUT/MODEL_OUTPUT/Mchl_complete.mat');
Mchl_complete=double(Mchl_complete_temp.Mchl_complete);
fprintf("\nMchl_complete data succesfully loaded")
fprintf('\n%s\n', repmat('-', 1, 45));
%
%-----EXTRACTING THE SINGULAR FIELDS FROM DATA
%
fprintf("Extracting the singular variables from data... \n");
Truedays=double(data.Truedays);
fprintf("Trueday has been extracted! \n");
Slon=double(data.Slon);
fprintf("Slon has been extracted! \n");
Slat=double(data.Slat);
fprintf("Slat has been extracted! \n");
Schl_complete=double(data.Schl_complete);
fprintf("Schl_complete has been extracted! \n");
%
%-----FIND DAYS WITH NO SATELLITE OBSERVATIONS-----
%   Unfortunatelly there seems to be some issue in
%   moving the satnan data from the python script
%   here.
%   Since the computation needed to find the satnan
%   data only needs the satellite chlorophylle values
%   (Schl) it is quickly done here once more to
%   ensure that no error ensues.
%   Furthermore, the removal of the "tempo" temporary
%   variables ensures that no extra space is 
%   actually occupied.
%
cnan=0;
for t = 1: Truedays
    tempo1(t)=1;
    tempo2=nansum(nansum(Schl_complete(t,:,:)));
    if tempo2==0
       cnan=cnan+1;
       tempo1=nan;
    end
end
%
string=strcat("The satnan index has been obtained!");
disp(string)
satnan=find(isnan(tempo1));
clear tempo1 tempo2
%
%-----MODEL LAND SEA MASK-----
%
MASK='C:/Tesi magistrale/Dati/mesh_mask.nc';
mask3d=nc_varget(MASK,'tmask');
mask3d=squeeze(mask3d);
%
%-----ELIMINATE DEGENERATE DIMENSION-----
%
Mmask=squeeze(mask3d(1,:,:));
%clear mask3d;
%
%-----FIND LAND GRIDPOINTS INDEXES FROM MODEL MASK-----
%
Mfsm=find(Mmask==0);
Mfsm_3d=find(mask3d==0);
%
%-----GET MODEL LAT & LON-----
%
Mlat=nc_varget(MASK,'nav_lat');
Mlon=nc_varget(MASK,'nav_lon');
%
%-----INITIALIZE AVERAGE TIME SERIES-----
%
%
M_TS_ave(1:Truedays)=nan;
S_TS_ave(1:Truedays)=nan;
%
if strcmp(chldlev,"l4")
%
%  -----IF "L4" SATELLITE DATAIS  INTERPOLATED ON MODEL GRID-----
%
%  ----GENERATE A LAND SEA MASK FOR SAT FIELDS-----
%
   tmp=squeeze(Schl_complete(1,:,:));
   satmask=abs(isnan(tmp)-1);
   Slat=Slat';
   clear tmp
%
%  ----ITERATE OVER DAYS-----
%
   string="Start interpolation loop";
   disp(string)
%
   for d=1:Truedays
%
      noflood=squeeze(Schl_complete(d,:,:));
%
%     -----EXPAND DATA OVER LAND-----
%
      flooded=Flood(noflood,5);
%
%     -----INTERPOLATE INTO MODEL GRID-----
%
%       Schl_interp(d,:,:)=interp2(Slon,Slat,flooded,Mlon,Mlat);
        Stmp=interp2(Slon,Slat,flooded,Mlon,Mlat);
%
%     ------MASK FIELDS-----
%
%       Schl_interp(d,:,:)=(squeeze(Schl_interp(d,:,:))).*Mmask(:,:);
        Smtp(Mfsm)=nan;
%        Schl_interp(d,:,:)=Stmp(:,:).*Mmask(:,:);
        Schl_interp(d,:,:)=Stmp(:,:);
%
       string=strcat("Interpolating day ", num2str(d));
       disp(string)
%
    end
%
    string="interpolation terminated";
    disp(string)
%
%
elseif strcmp(chldlev,"l3")
%
      Slat=Slat';

%-----IF "L3" MODEL DATA INTERPOLATED ON SATELLITE GRID-----
%
%     -----FIXING DIFFERENCE (in ROWS) BETWEEN MOD & SAT FIELDS----
%     
      MinMlat=min(min(Mlat));
      exSgrid=find(Slat <= MinMlat);
%      
      string="Interpolating the Model data onto the satellite grid";
      disp(string)
   for d = 1:Truedays
%
      string=strcat("Interpolating day ", num2str(d));
      disp(string)
%
%     ----GENERATE A LAND SEA MASK FOR SAT FIELDS-----
%
      Stmp=squeeze(Schl_complete(d,:,:));
      Stmp(exSgrid)=nan;
      outlierconc=15;
      outliers=find(Stmp>=outlierconc);
      Stmp(outliers)=nan;
      Schl_complete(d,:,:)=Stmp(:,:);
      satmask=abs(isnan(Stmp)-1);
      nobs=nansum(nansum(satmask));
      if nobs<=500
         satmask(:,:)=0;
      end
      satmasknan=find(satmask==0);
      Schl_complete(d,:,:)=Stmp(:,:);
      Schl_complete(satmasknan)=nan;
      clear Stmp
%
%     -----ELIMINATE DATA WITH NO SATELLITE OBSERVATIONS-----
%
      Mchl_complete(satnan,:,:)=nan;
      noflood=squeeze(Mchl_complete(d,:,:));
%
%     -----EXPAND DATA OVER LAND-----
%
      flooded=Flood(noflood,5);
%
%     -----INTERPOLATE INTO SATELLITE GRID-----
%
       Mtmp=interp2(Mlon,Mlat,flooded,Slon,Slat);
%
%     ------MASK FIELDS-----
%
       Mtmp(outliers)=nan;
       mgp=find(satmask==0);
       Mtmp(mgp)=nan;
       outliers=find(Mtmp>=outlierconc);
       Mtmp(outliers)=nan;
       clear mgp
       Mchl_interp(d,:,:)=Mtmp(:,:);
       clear Mtmp
%
   end 
   %-----SAVE AS NetCDF FILES-----
   fprintf("Setting the Output Folder...")
   interPath = 'C:/Tesi Magistrale/Dati/OUTPUT/INTERPOLATOR/';

   fprintf("\nSaving the Mchl_interp_l3.nc file...")
   ncwrite([interPath 'Mchl_interp_l3.nc'], 'Mchl_interp', Mchl_interp);
   fprintf("\nSaving the Schl_interp_l3.nc file...")
   ncwrite([interPath 'Schl_interp_l3.nc'], 'Schl_complete', Schl_complete);

   fprintf("Level 3 interpolated data has been saved!")
%
else
    string="Problem with chldlev!!!!";
    disp(string)
    return
%
end 
string="CHL data succesfully interpolated!";
disp(string)