### To read images 
# Location of the image dir
img_dir = 'nih/images-small/'
# Image variable consists of Images names as observations. Eg- '00025315_000.png','00020517_002.png','00009281_003.png', '00002898_002.png', '00025288_001.png'. 
# Extracting first observation in the Image Variable. 
sample_img = train_df.Image[0]  #Df.Var[0] 
plt.imread(os.path.join(img_dir, sample_img))




