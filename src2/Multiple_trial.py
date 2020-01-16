import face_recognition
import cv2
import pickle
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.475):
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
while ret:
    ret, frame1 = video1.read()
    rgb_frame1 = frame1[:, :, ::-1]
    ret, frame2 = video2.read()
    rgb_frame2 = frame2[:, :, ::-1]
    predictions1 = predict(rgb_frame1, model_path="trained_knn_model.clf")
    predictions2 = predict(rgb_frame2, model_path="trained_knn_model.clf")    
    for name, (top, right, bottom, left) in predictions1:
        cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame1, name, (left + 6, bottom - 6), font, 1.0, (255,255, 255), 1)
    
    for name, (top, right, bottom, left) in predictions2:
        cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame2, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame2, name, (left + 6, bottom - 6), font, 1.0, (255,255, 255), 1)
    
    cv2.imshow('Video1', frame1)
    cv2.imshow('Video2', frame2)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret = False

# Release handle to the webcam
video1.release()
video2.release()
cv2.destroyAllWindows()
