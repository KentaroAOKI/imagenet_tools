import json
import os
import pandas as pd
import requests

def getBingImages(search_word, offset = 0, count = 150):
    subscription_key = "put here your azure bing search api key"
    search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
    search_term = search_word
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": search_term, "mkt": "en-US", "safeSearch": "Moderate", "offset": str(offset), "count": str(count)}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results

def main():
    # image search parameters
    image_search_list = 'monster_list_en_utf8.csv'
    image_search_prefix = 'monster'
    download_dir = 'download_images'
    number_of_image = 1500


    # read list of search keyword
    search_words = pd.read_csv(image_search_list)
    search_words_names = search_words['Name']
    search_words_types = search_words['Type']

    # create log file for download status
    fp_image_list = open('download_images.csv', 'w')
    fp_image_list.writelines('name, number of image\n')

    for n in range(len(search_words_names)):
        query = search_words_names[n]
        dir_name = os.path.join(download_dir, search_words_names[n])
        dir_name = dir_name.replace(':','-')
        if (type(search_words_types[n]) is str):
            query = query + ' ' + search_words_types[n]
            dir_name = dir_name + '_' + search_words_types[n]
        
        # make directory for images
        isNewDir = False
        if (os.path.exists(dir_name) == False):
            os.makedirs(dir_name)
            isNewDir = True

        # get result of image search until "number_of_image" count
        thumbnail_urls = []
        for i in range(0,number_of_image,150):
            list_name = os.path.join(dir_name, 'images_{}.json'.format(i))
            if (os.path.isfile(list_name) == False):
                searched_images_json = getBingImages(image_search_prefix + ' ' + query, offset = i, count = 150)
                with open(list_name, 'w') as f:
                    json.dump(searched_images_json, f)
            with open(list_name, 'r') as f:
                image_datas = json.load(f)
            thumbnail_urls.extend([img["thumbnailUrl"] for img in image_datas["value"][:]])

        # write download status to log file
        fp_image_list.writelines('{}, {}\n'.format(query, len(thumbnail_urls)))

        # get images from search result
        if (isNewDir == True):
            image_no = 0
            for url in thumbnail_urls:
                try:
                    response = requests.get(url)
                    content_type = response.headers['Content-Type']
                    save_file = os.path.join(dir_name, str(image_no))
                    save_file = save_file + '.' + content_type.split('/')[1]
                    with open(save_file, 'wb') as saveFile:
                        print(save_file)
                        saveFile.write(response.content)
                    image_no = image_no + 1
                except:
                    print('cannot access {}.'.format(url))

    fp_image_list.close()

if __name__ == '__main__':
    main()
