import face_recognition
import cv2
import pickle
import get_points
import dlib

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.45):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_face_locations = face_recognition.face_locations(X_img_path,number_of_times_to_upsample=0)
    if len(X_face_locations) == 0:
       return []
    faces_encodings = face_recognition.face_encodings(X_img_path, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("UNKNOWN", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

video1 = cv2.VideoCapture(0)
video2 = cv2.VideoCapture(1)
ret = True


def track(img1,l1,name1,img2,l2,name2):
    tracker1 = dlib.correlation_tracker()
    tracker1.start_track(img1, dlib.rectangle(*l1[0]))
    tracker2 = dlib.correlation_tracker()
    tracker2.start_track(img1, dlib.rectangle(*l2[0]))
    ret = True
    while ret:
        ret, img1 = video1.read()
        ret, img2 = video2.read()
        
        tracker1.update(img1)
        tracker2.update(img2)
        
        rect1 = tracker1.get_position()
        rect2 = tracker2.get_position()
        
        pt11 = (int(rect1.left()), int(rect1.top()))
        pt21 = (int(rect1.right()), int(rect1.bottom()))
        
        pt12 = (int(rect2.left()), int(rect2.top()))
        pt22 = (int(rect2.right()), int(rect2.bottom()))

        cv2.rectangle(img1, pt11, pt21, (255, 255, 255), 3)
        cv2.rectangle(img2, pt12, pt22, (255, 255, 255), 3)

        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(img1, name1, (int(rect1.left()) + 6, int(rect1.bottom())- 6), font, 1.0, (0, 0, 255), 1)
        cv2.namedWindow("Image1", cv2.WINDOW_NORMAL)
    
        cv2.putText(img2, name2, (int(rect2.left()) + 6, int(rect2.bottom())- 6), font, 1.0, (0, 0, 255), 1)
        cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
    
        cv2.imshow("Image1", img1)
        cv2.imshow("Image2", img2)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            ret = False



while ret:
    ret, frame1 = video1.read()
    rgb_frame1 = frame1[:, :, ::-1]
    ret, frame2 = video2.read()
    rgb_frame2 = frame2[:, :, ::-1]
    predictions1 = predict(rgb_frame1, model_path="trained_knn_model.clf")
    predictions2 = predict(rgb_frame2, model_path="trained_knn_model.clf")    
    for name1, (top, right, bottom, left) in predictions1:
        (x,y,w,h)=predictions1[0][1]
        l1 = [(h,x,y,w)]   
    
    for name2, (top, right, bottom, left) in predictions2:
        (x,y,w,h)=predictions2[0][1]
        l2 = [(h,x,y,w)]     
    track(frame1,l1,name1,frame2,l2,name2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret = False

# Release handle to the webcam
video1.release()
video2.release()
cv2.destroyAllWindows()
