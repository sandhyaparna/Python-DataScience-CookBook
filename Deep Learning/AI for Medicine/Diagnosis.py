### To read images
# Image variable consists of Images names as observations.Eg- '00025315_000.png','00020517_002.png','00009281_003.png', '00002898_002.png', '00025288_001.png'. 
# Extract Image names
ImageNames = Df['Image'].values

# Location of the image dir
img_dir = 'nih/images-small/'
# Extracting first observation in the Image Variable. 
sample_img = Df.Image[0]  #Df.Var[0] 
# Extraing First image
plt.imread(os.path.join(img_dir, sample_img)) #Gives 2-D array of pixels
image = plt.imread(os.path.join(img_dir, sample_img))  #
plt.imshow(image,cmap='gray')
plt.axis('off') # Removes Axis and Graph like lines
plt.tight_layout()

# Extract numpy values from Image column in data frame
images = train_df['Image'].values
# Extract 9 random images from it
random_images = [np.random.choice(images) for i in range(9)]
# Location of the image dir
img_dir = 'nih/images-small/'
print('Display Random Images')
# Adjust the size of your images
plt.figure(figsize=(20,10))
# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off') 
# Adjust subplot parameters to give specified padding
plt.tight_layout() 

