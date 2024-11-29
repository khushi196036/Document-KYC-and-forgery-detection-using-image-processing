# match.py

import face_recognition
import numpy as np

def match_faces(image_path1, image_path2):
    print("match.py is called")
    # Load images containing faces
    image1 = face_recognition.load_image_file(image_path1)
    image2 = face_recognition.load_image_file(image_path2)

    # Find face encodings for both images
    face_encoding1 = face_recognition.face_encodings(image1)
    face_encoding2 = face_recognition.face_encodings(image2)

    # Check if at least one face was found in each image
    if len(face_encoding1) > 0 and len(face_encoding2) > 0:
        # Convert face encodings to numpy arrays
        face_encoding1 = np.array(face_encoding1[0])  # Assuming there's only one face in each image
        face_encoding2 = np.array(face_encoding2[0])

        # Calculate the face distance (lower is better)
        face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)

        # Set a matching threshold (you can adjust this threshold)
        matching_threshold = 0.6
        # Check if the face distance is below the threshold
        if face_distance[0] < matching_threshold:
            return True  #"Match successful"
        else:
            return False    #"Match unsuccessful"
    else:
        return "No face found in one or both images"
