from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
model = load_model('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/dataset/subweights.best.image_classifier.hdf5')
test_image = image.load_img('C:/Users/abd/OneDrive/Bureau/Projet_FIN/src/assets/images/upload/phone.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)

if result[0][0] == result.max(axis=1): 
    prediction = 'Android'
elif result[0][1] == result.max(axis=1):
    prediction = 'Apparel Set'
elif result[0][2] == result.max(axis=1):
    prediction = 'Apple'
elif result[0][3] == result.max(axis=1): 
    prediction = 'Bags Accessories'
elif result[0][4] == result.max(axis=1):
    prediction = 'Bags Sporting'
elif result[0][5] == result.max(axis=1):
    prediction = 'Bath Body'
elif result[0][6] == result.max(axis=1): 
    prediction = 'Belts'
if result[0][7] == result.max(axis=1): 
    prediction = 'Bottomwear'
elif result[0][8] == result.max(axis=1):
    prediction = 'Children'
elif result[0][9] == result.max(axis=1):
    prediction = 'Classics'
elif result[0][10] == result.max(axis=1): 
    prediction = 'Crime'
elif result[0][11] == result.max(axis=1):
    prediction = 'Cufflinks'
elif result[0][12] == result.max(axis=1):
    prediction = 'Dress'
elif result[0][13] == result.max(axis=1): 
    prediction = 'Eyes'
if result[0][14] == result.max(axis=1): 
    prediction = 'Eyewear Accessories'
elif result[0][15] == result.max(axis=1):
    prediction = 'Eyewear Sporting'
elif result[0][16] == result.max(axis=1):
    prediction = 'Families Relationship'
elif result[0][17] == result.max(axis=1): 
    prediction = 'Fantasy'
elif result[0][18] == result.max(axis=1):
    prediction = 'Flip Flops'
elif result[0][19] == result.max(axis=1):
    prediction = 'Fragrance'
elif result[0][20] == result.max(axis=1): 
    prediction = 'General'
if result[0][21] == result.max(axis=1): 
    prediction = 'Gift'
elif result[0][22] == result.max(axis=1):
    prediction = 'Gloves'
elif result[0][23] == result.max(axis=1):
    prediction = 'Hair'
elif result[0][24] == result.max(axis=1): 
    prediction = 'Headwear Accessories'
elif result[0][25] == result.max(axis=1):
    prediction = 'Headwear Sporting'
elif result[0][26] == result.max(axis=1):
    prediction = 'Honor'
elif result[0][27] == result.max(axis=1): 
    prediction = 'Huawei'
if result[0][28] == result.max(axis=1): 
    prediction = 'Innerwear'
elif result[0][29] == result.max(axis=1):
    prediction = 'Jewellery'
elif result[0][30] == result.max(axis=1):
    prediction = 'Lips'
elif result[0][31] == result.max(axis=1): 
    prediction = 'Literature'
elif result[0][32] == result.max(axis=1):
    prediction = 'Loungewear Nightwear'
elif result[0][33] == result.max(axis=1):
    prediction = 'Makeup'
elif result[0][34] == result.max(axis=1): 
    prediction = 'Nails'
if result[0][35] == result.max(axis=1): 
    prediction = 'Nokia'
elif result[0][36] == result.max(axis=1):
    prediction = 'Oneplus'
elif result[0][37] == result.max(axis=1):
    prediction = 'Oppo'
elif result[0][38] == result.max(axis=1): 
    prediction = 'Others'
elif result[0][39] == result.max(axis=1):
    prediction = 'Redmi'
elif result[0][40] == result.max(axis=1):
    prediction = 'Romance'
elif result[0][41] == result.max(axis=1): 
    prediction = 'Samsung'
if result[0][42] == result.max(axis=1): 
    prediction = 'Sandal'
elif result[0][43] == result.max(axis=1):
    prediction = 'Saree'
elif result[0][44] == result.max(axis=1):
    prediction = 'Scarves'
elif result[0][45] == result.max(axis=1): 
    prediction = 'Science Fiction'
elif result[0][46] == result.max(axis=1):
    prediction = 'Shoe Accessories'
elif result[0][47] == result.max(axis=1):
    prediction = 'Shoes'
elif result[0][48] == result.max(axis=1): 
    prediction = 'Short Stories'
if result[0][49] == result.max(axis=1): 
    prediction = 'Skin Care'
elif result[0][50] == result.max(axis=1):
    prediction = 'Socks'
elif result[0][51] == result.max(axis=1):
    prediction = 'Sony'
elif result[0][52] == result.max(axis=1): 
    prediction = 'Sport Equipement'
elif result[0][53] == result.max(axis=1):
    prediction = 'Stoles'
elif result[0][54] == result.max(axis=1):
    prediction = 'Ties'
elif result[0][55] == result.max(axis=1): 
    prediction = 'Topwear'
if result[0][56] == result.max(axis=1): 
    prediction = 'Vivo'
elif result[0][57] == result.max(axis=1):
    prediction = 'Wallets'
elif result[0][58] == result.max(axis=1):
    prediction = 'Watches Accessories'
elif result[0][59] == result.max(axis=1): 
    prediction = 'Watches Sportings'
elif result[0][60] == result.max(axis=1):
    prediction = 'Water Bottle'
elif result[0][61] == result.max(axis=1):
    prediction = 'Young Adults'
print(prediction)