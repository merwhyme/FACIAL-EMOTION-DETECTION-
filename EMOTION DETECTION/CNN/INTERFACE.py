import streamlit as st
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
face_classifier = cv2.CascadeClassifier(r"C:\Users\jamal\Downloads\haarcascade_frontalface_default.xml")

if face_classifier.empty():
    print("Erreur: Impossible de charger le classificateur en cascade")

classifier = load_model(r"C:\\Users\\jamal\\Downloads\\model.h5")



emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']



# Fonction pour détecter les émotions en temps réel à partir de la caméra
def detect_emotion():
    st.title("Détection d'émotion en temps réel")
    st.write("Ouvrez la caméra pour détecter les émotions en temps réel.")

    # Démarrer la capture vidéo
    cap = cv2.VideoCapture(0)

    # Boucle pour la détection en temps réel
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Fonction pour la section "À propos de nous"
def about_us():
    st.title("À propos de nous")
    st.write("Nous sommes une équipe d'étudiants passionnés en Big Data et Intelligence Artificielle à l'École Nationale des Sciences Appliquées de Tétouan (ENSA). Notre parcours académique nous a plongés dans les profondeurs de l'analyse de données volumineuses et des techniques d'intelligence artificielle, nous préparant ainsi à relever les défis de l'ère numérique.")
    st.write("Notre intérêt pour ces domaines novateurs nous a amenés à explorer différentes applications et solutions, allant de l'analyse prédictive à la reconnaissance d'émotions en passant par la modélisation de systèmes complexes. Grâce à notre formation rigoureuse et notre curiosité sans limite, nous nous efforçons de repousser les limites de la technologie et de trouver des solutions innovantes pour les problèmes contemporains.")
    st.write("Dans le cadre de notre projet de ML, nous avons développé MyEmotion, une application de reconnaissance d'émotions en temps réel. Cette application combine habilement les puissants outils d'OpenCV avec les réseaux neuronaux convolutionnels pour identifier rapidement et précisément les expressions faciales. Avec une interface conviviale, il vous suffit d'utiliser votre caméra pour capturer une vidéo en direct. Nos algorithmes analysent rapidement les expressions faciales, reconnaissant une large gamme d'émotions, notamment la joie, la tristesse, la colère, la surprise, la peur et le dégoût.")
    st.write("Nous sommes fiers de présenter MyEmotion, fruit de notre engagement, de notre travail d'équipe et de notre passion pour l'intelligence artificielle. Nous espérons que cette application vous fournira une expérience enrichissante et démontrera le potentiel transformateur de la technologie dans notre quotidien.")
    
    
   
    
   
    # Définition de la page principale de l'application
def main():
    st.sidebar.title("Menu")
    menu_options = ["Accueil", "Détection d'émotion", "À propos de nous"]
    choice = st.sidebar.radio("Navigation", menu_options)

    if choice == "Accueil":
        
        st.image(r"C:\Users\jamal\Downloads\WhatsApp Image 2024-05-06 à 23.59.39_418379d9.jpg", width=500)

        st.title("Bienvenue sur MyEmotion.")
        st.write("Notre application MyEmotion de reconnaissance d'émotions, qui combine OpenCV et les réseaux neuronaux convolutionnels pour identifier rapidement les émotions en temps réel. Avec une interface conviviale, il vous suffit d'utiliser votre caméra pour capturer une vidéo en direct. Nos algorithmes analysent rapidement les expressions faciales, reconnaissant une large gamme d'émotions, notamment la joie, la tristesse, la colère, la surprise, la peur et le dégoût.")
        
        st.markdown('<span style="font-size:20px;">Utilisez le menu sur le côté pour naviguer.</span>', unsafe_allow_html=True)



    elif choice == "Détection d'émotion":
        detect_emotion()

    elif choice == "À propos de nous":
        about_us()


if __name__ == "__main__":
    main()



