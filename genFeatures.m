function genFeatures
%% For each input image, generate and save the "36 layer features" of MultiScales (4), MultiCropped(3x3) and flipped(2)
global  net

addpath('/path/to/matconvnet-1.0-beta17/matlab');                           %path to matconvnet matlab folder
modelPath = '/path/to/matconvnet-1.0-beta17/data/models/vgg-face.mat';      %path to vgg_face weigths  

run  vl_setupnn

genANDsave= 1; %flag to generate the Deep features and save them
visualize = 0; %flag to show the cropped images on which to extract the feautres
net= load(modelPath);
net = vl_simplenn_tidy(net);
disp('net loaded...');

net.meta.normalization.imageSize(1)

pathname = '/path/to/data';     %path to data folder (lfw funneled or AR)
pathname_OUT = '/path/to/out/';          %pah to output folder


SCALES =  300:50:450;
nSCALES = length(SCALES);

%center displacements:
rDisplacement = -10:10:10;  %row displacement
cDisplacement = -10:10:10; %column displacement
nCROPS =  size(rDisplacement,2) + size(cDisplacement,2);

DIRS = dir(pathname);
count = 0;
disp(length(DIRS));
for i=3:length(DIRS)
    subDir = dir([pathname '/' DIRS(i).name '/*.jpg']);
    if length(subDir)>=10 %&& ~exist([pathname '-DeepFeats/' DIRS(i).name], 'dir')
        
        %nImgProc =  min([length(subDir), 10]);
        nImgProc = length(subDir);          %take all the images for the subjects with at least 10 images
        count = count +1;
        disp(count);
        
        for j=1: nImgProc
            clear DeepFeat36_mS_mC_2F;
            Sdim = net.meta.normalization.imageSize(1);
            ASdim = floor(net.meta.normalization.imageSize(1)/2);
           
            nameOUT = [pathname_OUT  DIRS(i).name '/' subDir(j).name(1:end-4) '_DeepFeat36_LargeScales_MultiCrops_MultiFlips.mat'];
                    
            if ~exist(nameOUT, 'file')
                im = imread([pathname '/' DIRS(i).name '/' subDir(j).name]);
                for scale=1:size(SCALES,2)
                    img_scaled = imresize(im, [SCALES(scale) SCALES(scale)]);
                    [rIm, cIm, ~] = size(img_scaled);
                    rCenter = floor(rIm/2);
                    cCenter = floor(cIm/2);
                    crop = 0;
                    for rCrop=1:size(rDisplacement,2)
                        for cCrop=1:size(cDisplacement,2)
                            crop=crop + 1;
                            rStart= rCenter + rDisplacement(rCrop)*scale - ASdim +1;
                            rEnd = rCenter + rDisplacement(rCrop)*scale +  ASdim;
                            cStart = cCenter + cDisplacement(cCrop)*scale - ASdim +1;
                            cEnd = cCenter + cDisplacement(cCrop)*scale + ASdim;
                            if cStart>0 && cEnd<cIm && rStart>0 && rEnd<rIm
                                img_cropped = img_scaled(rStart: rEnd, cStart:cEnd, :);
                                
                                for flip=1:2
                                    if flip == 1
                                        img_flip = img_cropped;
                                    else
                                        img_flip = fliplr(img_cropped);
                                    end
                                    
                                    featExtract = img_flip;
                                     
                                    if visualize &&  flip==1
                                        imshow(featExtract);
                                        %imwrite(img_flip, ['S' num2str(scale) '_C' num2str(crop) '.png']);
                                        title(['Scale: ' num2str(scale) ', Crop: [' num2str(cDisplacement(cCrop))  ',' num2str(rDisplacement(rCrop)) '], Flip: ' num2str(flip)]);
                                        pause
                                    end
                                    
                                    if genANDsave
                                        DeepFeat36_mS_mC_2F{scale}{crop}{flip} = genF(featExtract);
                                    end    
                                end
                            else
                                disp('out of Scale!');
                            end
                        end
                    end
                end
                
                if genANDsave
                    dirFile =  [pathname_OUT  DIRS(i).name '/'];
                    if ~exist(dirFile, 'dir')
                        mkdir(dirFile);
                    end
                    disp(count); 
                  
                    save( [pathname_OUT  DIRS(i).name '/' subDir(j).name(1:end-4) '_DeepFeat36_LargeScales_MultiCrops_MultiFlips.mat'],'DeepFeat36_mS_mC_2F', 'nCROPS', 'nSCALES');
                end
            else
                disp('Exist already');
            end
        end 
    end
end
end


function Feat = genF(img)
global net

im_ = single(img);
im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage);
res = vl_simplenn(net, im_) ;
Feat = squeeze((res(36).x));

end
