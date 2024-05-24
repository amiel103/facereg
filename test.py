import cv2
import numpy as np
from PIL import Image

from imgbeddings import imgbeddings

# Initialize Imgbeddings model
imgbeddings_model = imgbeddings()

reference_images = {
    "Carmiegildo Egot": "embd/megmeg.jpg",
    "Daryl Clark Bayotas": "embd/dadars.jpg",
    "Jeborg Pagaling": "embd/jeborg.jpg",
    "Roque Jay Tagam": "embd/tagam.jpg"
}

def load_and_embed_image(image_path, model):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_resize = cv2.resize(image_rgb, (200, 200))

    # print(image_resize)


    image = Image.fromarray(image_rgb)
    embedding = model.to_embeddings(image)
    print('loaded',image.size)
    return embedding

# Load reference images and compute embeddings

reference_embeddings = {}
for name, img_path in reference_images.items():
    reference_embeddings[name] = load_and_embed_image(img_path, imgbeddings_model)


# print(reference_embeddings)
def cosine_similarity(embedding1, embedding2):
    embedding1 = np.squeeze(np.asarray(embedding1))

    embedding2 = np.squeeze(np.asarray(embedding2))
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera feed
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # time.sleep(2.5)

        face_img = frame[y:y+h, x:x+w]
        image_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image_rgb, (200, 200))
        im_pil = Image.fromarray(image_rgb)
        # print('captured' , im_pil.size)
        captured_embedding = imgbeddings_model.to_embeddings(im_pil)

        for name, ref_embedding in reference_embeddings.items():
            similarity = cosine_similarity(captured_embedding, ref_embedding)
            threshold = 0.9  # Set a similarity threshold
            print(name , ' - ' , similarity)
            if similarity > threshold:
                print(name , ' - ' , similarity)
                # attendance[name] = "Present"
            else:
                # attendance[name] = "Absent"
                print('unidentified')

        # print(captured_embedding)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        

    
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()