function tot=flood(matrice_in,loop)

% Author: Luca Giacomelli 2009

% amplio il dominio di 1 punto
idx{1}   = (1:size(matrice_in,1)) + 1;
idx{2}   = (1:size(matrice_in,2)) + 1;
dummy = repmat(9999, size(matrice_in)+2);
dummy(idx{:}) = matrice_in;
% trovo i nan
idx=find(isnan(dummy));

% metto a nan anche i bordi
dummy(dummy==9999)=nan;

% matrice dei vicini che voglio analizzare
M=size(dummy,1);
neighbor_offsets = [M, M+1, 1, -M+1, -M, -M-1, -1, M-1]';

for count=1:loop
    if isempty(idx)
        break
    end
    neighbors=idx(:,ones(8,1))'+neighbor_offsets(:,ones(size(idx,1),1));
    
    % esplicito nanmean
    % media=nanmean(dummy(neighbors));
    mat=dummy(neighbors);
    nans=isnan(mat);
    mat(nans)=0;
    snn=sum(nans==0);
    media=sum(mat)./snn;
    
    dummy(idx)=media;
    idx(snn>0)=[];
% % %     imagesc(dummy)
% % %     pause(1)
end
tot=dummy(2:end-1, 2:end-1);

