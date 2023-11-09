import tkinter
import cv2
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk





def image_detect():
    def highlightFace(net, frame, conf_threshold=0.7,x=255,y=255,z=255):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (x,y,z), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes


    def get_filename(file_path):
        return os.path.basename(file_path)
     
    def select_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            filename = get_filename(file_path)
        return filename
    
    args=select_image()

    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    video=cv2.VideoCapture(args if args else 0)
    padding=20
    while cv2.waitKey(1)<0 :
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")
            break

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
            if f'{gender}'== 'Male':
                resultImg,faceBoxes=highlightFace(faceNet,frame,0.7,216, 180,0)
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (216, 180,0), 2, cv2.LINE_AA)
                
            else :
                resultImg,faceBoxes=highlightFace(faceNet,frame,0.7,196, 102 , 255)
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (196, 102 , 255), 2, cv2.LINE_AA)
                
            cv2.imshow("Detecting age and gender", resultImg)



def camera_launch():
    def detectFace(net,frame,confidence_threshold=0.7):
        frameOpencvDNN=frame.copy()
        print(frameOpencvDNN.shape)
        frameHeight=frameOpencvDNN.shape[0]
        frameWidth=frameOpencvDNN.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>confidence_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(255,255,255),int(round(frameHeight/150)),8)
        return frameOpencvDNN,faceBoxes
            
        
    faceProto='opencv_face_detector.pbtxt'
    faceModel='opencv_face_detector_uint8.pb'
    ageProto='age_deploy.prototxt'
    ageModel='age_net.caffemodel'
    genderProto='gender_deploy.prototxt'
    genderModel='gender_net.caffemodel'

    genderList=['Male','Female']
    ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    video=cv2.VideoCapture(0)
    padding=20
    while cv2.waitKey(1)<0:
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break
            
        resultImg,faceBoxes=detectFace(faceNet,frame)
        
        if not faceBoxes:
            print("No face detected")
            tkinter.messagebox.showerror("Erreur","No face detected")
            break
        
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            if f'{gender}'== 'Male':
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (216, 180,0), 2, cv2.LINE_AA)
            else :
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (196, 102 , 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and Gender",resultImg)
            
            
            if cv2.waitKey(1) == ord('a'):
                break
    cv2.destroyAllWindows()



    


root = ctk.CTk()
ctk.set_appearance_mode("system")
root.title('A&G')

#les frames
frame = ctk.CTkFrame(root,fg_color="transparent",corner_radius = 10)
frame.pack(pady=10)


#icon_emsi
image_emsi= Image.open("tstk/emsi.png")
image_emsi = image_emsi.resize((400,100))
photo = ImageTk.PhotoImage(image_emsi)
label_emsi = ctk.CTkLabel(frame,text =" ",width=400,image=photo,anchor="w")
label_emsi.pack(padx=30,side="left")

#icon_honoris
image_honoris= Image.open("tstk/honoris.png")
image_honoris = image_honoris.resize((300,100))
photo = ImageTk.PhotoImage(image_honoris)
label_honoris = ctk.CTkLabel(frame,text=" ",width=400,image=photo,anchor="e")
label_honoris.pack(padx=30,side="right")

#pfa
label_projet=ctk.CTkLabel(frame, text="\nProjet de Fin d'Année\n3éme Année Ingenieurie Informatique et Réseaux groupe 7\n",
                   font= ('FONT_HERSHEY_SIMPLEX',20,"bold"),
                   
                  )
label_projet.pack()

#realise par 
label_e=ctk.CTkLabel(frame, text="Realisé par : Asmae MOUBARRIZ\n                 Achraf AKRACHE",
                   font= ('FONT_HERSHEY_SIMPLEX',20),
                  )
label_e.pack()

#encadre par 
label_p=ctk.CTkLabel(frame, text="Encadré par : Prof. Nisrine DAD    ",
                   font= ('FONT_HERSHEY_SIMPLEX',20),
                  )
label_p.pack()



#label_titre
label=ctk.CTkLabel(root, text="Détection d'Âge et du Genre !",
                   corner_radius = 10,
                   font= ('FONT_HERSHEY_SIMPLEX',40)
                  )
label.pack(pady=20,padx=30)


#icon_application
image= Image.open("tstk/facia.png")
image = image.resize((300, 300))
photo = ImageTk.PhotoImage(image)
label = ctk.CTkLabel(root,text=" ", image=photo,corner_radius = 10,)
label.pack()





#button de detection d'image 
button_image=ctk.CTkButton(root,text="Détecter l'âge et le genre",
                    font = ('FONT_HERSHEY_SIMPLEX',25,"bold"),
                    command = image_detect,
                    width=350,
                    corner_radius = 10,
                    hover_color="green" )
button_image.pack(pady=15)

#button de detection de camera  
camera=ctk.CTkButton(root,text="Ouvrir la camera",
                    font= ('FONT_HERSHEY_SIMPLEX',25,"bold"),
                    command= camera_launch,
                    width=350,
                    corner_radius = 10,
                    hover_color="green" )
camera.pack(pady=15)


#button de quitter 
quitter=ctk.CTkButton(root, text="Quitter",
                      font= ('FONT_HERSHEY_SIMPLEX',25,"bold"),
                      command=root.destroy,
                      corner_radius = 10,
                      width=350,
                      hover_color="green" )
quitter.pack(pady=15)


#camera launch label
label_quitter=ctk.CTkLabel(root,text="Appuyer sur A pour arrêter la camera",
                           corner_radius = 10,
                           font= ('FONT_HERSHEY_SIMPLEX',25,"bold"))
label_quitter.pack(pady=15)




root.mainloop()