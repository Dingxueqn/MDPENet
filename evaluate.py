from PIL import Image
import os
import json


def get_vaslue(predict_folders_path, label_folders_path):
    predict_folders = os.listdir(predict_folders_path)
    label_folders = os.listdir(label_folders_path)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for folder in predict_folders:
        predict_folder_path = os.path.join(predict_folders_path, folder)
        label_folder_path = os.path.join(label_folders_path, folder)
        predict = Image.open(predict_folder_path)
        predict = predict.convert('RGBA')
        label = Image.open(label_folder_path)
        label = label.convert('RGBA')
        heigh, width = predict.size
        # save_name = str(folder).split('.')[0]
        for i in range(heigh):
            for j in range(width):
                r_1, g_1, b_1, a_1 = predict.getpixel((i, j))
                r_2, g_2, b_2, a_2 = label.getpixel((i, j))
                if r_1 == 255:
                    if r_2 == 255:
                        TP += 1
                    if r_2 == 0:
                        FP += 1
                if r_1 == 0:
                    if r_2 == 255:
                        FN += 1
                    if r_2 == 0:
                        TN += 1
    return float(TP), float(TN), float(FP), float(FN)


def list2txt(list, save_path, txt_name):
    with open(save_path + r'/' + txt_name, 'w') as f:
        json.dump(list, f)


def evoluation(TP, TN, FP, FN):
    evo = []
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # miou
    miou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    # F1
    f1 = 2 * ((precision * recall) / (precision + recall))
    evo.append('accuracy:{}  precision:{}  recall:{}  miou:{} f1:{}'.format(accuracy, precision, recall, miou, f1))
    print(evo)
    return evo


if __name__ == '__main__':
    label_path = "output/colon/gt"  
    predict_path = "output/colon/pred" 
    TP, TN, FP, FN = get_vaslue(predict_path, label_path)
    evoluation(TP, TN, FP, FN)

