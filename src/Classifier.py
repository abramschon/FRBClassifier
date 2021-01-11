import numpy as np
from tensorflow import keras
from Data import process_images

def main():

    data_dir = input("Enter path to directory of images:\n").strip()
    if len(data_dir) == 0:
        data_dir = "example_data/test_data" #by default, use data in test_data folder

    # process new data 
    ds = process_images(data_dir, prop=1, mean=0.54, sd=0.14)

    # get ids and images
    ids_ds = ds.map(lambda image, id: id)
    ids = next(iter(ids_ds)).numpy().astype('U')
    images = ds.map(lambda image, id: image)

    # load model
    model = keras.models.load_model('tfmodel')
    # model.summary()
    
    output = input("Enter name of file to save predictions (default is 'predictions.csv'):\n").strip()
    if len(output) == 0:
        ouput = 'predictions.csv'
        
    # write predictions out
    prob = model.predict(images).flatten()
    pred = np.round(prob)

    np.savetxt(
        ouput,
        np.rec.fromarrays([ids, prob, pred]),
        fmt=['%s', '%f', '%f'],
        delimiter=',',
        header='id,prob,pred',
        comments='',
    )

    print("Done")

if __name__ == "__main__":
    main()