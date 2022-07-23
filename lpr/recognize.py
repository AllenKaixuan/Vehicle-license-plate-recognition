import cv2 as cv

from lpr import predict,segmentation, hsv_plate_locator
#import predict
#import segmentation
#import hsv_plate_locator


def recognize(image):
    # locate
    plates_image = hsv_plate_locator.locate(image)
    for i in range(len(plates_image)):
        cv.imwrite("temp/plate"+str(i)+".jpg", plates_image[i])
    # segment
    seg = segmentation.Segmentation()
    chars_image = seg.segment(plates_image)
    # pridict
    char_result = []
    confidence_result = []
    for i in range(len(chars_image)):
        char_output = []
        confidence_output = []
        # ch-pridict
        char, confidence = predict.predict_ch(chars_image[i][0])
        char_output += char
        confidence_output.append(confidence)
        cv.imwrite("temp/plate"+str(i)+"_0.jpg", chars_image[i][0])
        # en-pridict
        for j in range(1, min(len(chars_image[i]), 7)):
            char, confidence = predict.predict_en(chars_image[i][j])
            char_output += char
            confidence_output.append(confidence)
            cv.imwrite("temp/plate"+str(i)+"_"+str(j)+".jpg",
                       chars_image[i][j])
        if len(char_output) >= 4:
            char_result.append(''.join(char_output))
            confidence_result.append(confidence_output)

    return char_result, confidence_result


if __name__ == "__main__":
    image_path = './cars/plate8.jpg'
    image = cv.imread(image_path)
    char_result, confidence_result = recognize(image)
    print(char_result)
    print(confidence_result)
