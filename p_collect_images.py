import os
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36  # 26 letters (A-Z) + 10 digits (0-9)
dataset_size = 200

# Mapping: 0-25 = A-Z, 26-35 = 0-9
def get_class_name(class_num):
    if class_num < 26:
        return chr(65 + class_num)  # A-Z
    else:
        return str(class_num - 26)  # 0-9

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    class_name = get_class_name(j)
    print('Collecting data for class {} ({})'.format(j, class_name))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready for "{}". Press "Q" ! :)'.format(class_name), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()