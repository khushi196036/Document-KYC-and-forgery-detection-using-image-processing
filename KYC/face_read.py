import cv2
import matplotlib.pyplot as plt

image = cv2.imread("temp.jpg")
coords=[]
face_id=0

def generate_Dataset(img ,face_id):
    cv2.imwrite(f"data/user_{face_id}.jpg", img)
    # cv2.imwrite("data/user.jpg",img)


def detact_f():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        coords = [x, y, w, h]

    if len(coords) == 4:
        roi_image = image[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        global face_id
        face_id +=1
        generate_Dataset(roi_image,face_id)
    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Faces')
    plt.axis('off')  # Hide the axis
    plt.show()

detact_f()
