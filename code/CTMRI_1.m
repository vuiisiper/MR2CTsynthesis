disp('Enter 1 for CT, 2 for MR and 3 for MRCT');
prompt = 'Insert number: '
x = input(prompt);
%
if x == 1
    mod = 'VCT';
    VCT = niftiread('CT-MRI-5634_CT_norm.nii');
elseif x ==2
    mod = 'VMR';
    VMR = niftiread('CT-MRI-5634_MR_norm.nii');
else
    mod = 'VMRCT';
    VMRCT = niftiread('CT-MRI-5634_MR_norm_lab_2_.nii');
end

if x == 1
    [sag, cor, axial] = size(VCT);  %row, column and number of slices
elseif x ==2
    [sag, cor, axial] = size(VMR);
else
    [sag, cor, axial] = size(VMRCT);
end

t = 1;
while t<10
    t=t+1;
    for i = 1:axial
        if x == 1            
            imagesc(squeeze(VCT(:,:,i)));
        elseif x == 2
            imagesc(squeeze(VMR(:,:,i)));
        else
            imagesc(squeeze(VMRCT(:,:,i)));
        end
    if (i == 1)
        cLim_v = get(gca, 'CLim');
    else
        set(gca, 'CLim', cLim_v)
    end
    axis image
    axis on
    colormap(gray)
if x == 1    
    title(['CT data: ', num2str(i)]);
elseif x ==2
    title(['MR data: ', num2str(i)]);
else
    title(['MR/CT data: ', num2str(i)]);
end
    drawnow
    mov(i) = getframe;
  
    end
end
%% CT image voxel
voxCT = V(117,144,100)
%% MR image voxel
voxMR = V(151,145,91)
%%
i = 100;
% for nSlice = 1:axial 
  for nCols = 1:cor
    for nRows = 1:sag
        plane(nRows,nCols) = V(nRows,nCols,i); 
    end
  end
%% Converting .gz to .nii
imds = imageDatastore(cd,'IncludeSubfolders',true,'LabelSource','FolderNames','FileExtensions','.gz');
nGZ = numel(imds.Files);
for i = 1:nGZ
    gunzip(imds.Files{i});
    clear imds.Files{i};
end
