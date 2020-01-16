import face_recognition
import cv2
import pickle
import dlib
import get_points
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
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
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def equi(x1,y1,x2,y2):
    return (((x1-x2)**2)+((y1-y2)**2))**0.5


video1 = cv2.VideoCapture(0)
video2 = cv2.VideoCapture(1)
ret = True
l1 =[]
nae1 = []
l2 =[]
nae2 = []

def track(img1,l1,nae1,img2,l2,nae2):
    tracker1 = [dlib.correlation_tracker() for _ in range(len(l1))]
    [tracker1[i].start_track(img1, dlib.rectangle(*rect)) for i, rect in enumerate(l1)]
    tracker2 = [dlib.correlation_tracker() for _ in range(len(l2))]
    [tracker2[i].start_track(img2, dlib.rectangle(*rect)) for i, rect in enumerate(l2)]    
    ret = True
    if "unknown" in nae1:
        return True
    while ret:
        la1 = []
        na1 = []
        la2 = []
        na2 = []

        ret, img1 = video1.read()
        rgb_frame1 = img1[:, :, ::-1]
        ret, img2 = video2.read()
        rgb_frame2 = img2[:, :, ::-1]
        
        predictions1 = predict(rgb_frame1, model_path="trained_knn_model.clf")    
        for name, (x,y,w,h) in predictions1:
            la1.append((h,x,y,w))
            na1.append(name)

        predictions2 = predict(rgb_frame2, model_path="trained_knn_model.clf")    
        for name, (x,y,w,h) in predictions2:
            la2.append((h,x,y,w))
            na2.append(name)
        
        for j in na1:
            if j == "unknown":
                continue
            if j not in nae1:
                dist = []
                (L,T,R,B) = la1[na1.index(j)]
                for k in range(len(l1)):
                    (lef,top,rig,bot) = l1[k]
                    dist.append(equi(L,T,lef,top))
                if min(dist)<75.0:
                    continue
                else:
                    nae1.append(j)
                    l1.append(la1[na1.index(j)])
                    tracker1 = [dlib.correlation_tracker() for _ in range(len(l1))]
                    [tracker1[i].start_track(img1, dlib.rectangle(*rect)) for i, rect in enumerate(l1)]
        
        for j in na2:
            if j == "unknown":
                continue
            if j not in nae2:
                dist = []
                (L,T,R,B) = la2[na2.index(j)]
                for k in range(len(l2)):
                    (lef,top,rig,bot) = l2[k]
                    dist.append(equi(L,T,lef,top))
                if min(dist)<75.0:
                    continue
                else:
                    nae2.append(j)
                    l2.append(la2[na2.index(j)])
                    tracker2 = [dlib.correlation_tracker() for _ in range(len(l2))]
                    [tracker2[i].start_track(img2, dlib.rectangle(*rect)) for i, rect in enumerate(l2)] 

        i = 0   
        for i in range(len(tracker1)):
            m,n,_ = img1.shape
            tracker1[i].update(img1)
            rect = tracker1[i].get_position()
            l1[i] = (int(rect.left()), int(rect.top()),int(rect.right()), int(rect.bottom()))
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(img1, pt1, pt2, (0,0, 255), 3)
            cv2.putText(img1, nae1[i], (int(rect.left()) + 6, int(rect.bottom()) - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
            print("Object {} tracked at [{}, {}] \r".format(i, pt1, pt2))
            
        i = 0
        for i in range(len(tracker2)):
            tracker2[i].update(img2)
            rect = tracker2[i].get_position()
            l2[i] = (int(rect.left()), int(rect.top()),int(rect.right()), int(rect.bottom()))
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(img2, pt1, pt2, (0,0, 255), 3)
            cv2.putText(img2, nae2[i], (int(rect.left()) + 6, int(rect.bottom()) - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
            print("Object {} tracked at [{}, {}] \r".format(i, pt1, pt2))
        
        cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera 1", img1)
        cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera 2", img2)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break
            return False

while ret:
    ret, frame1 = video1.read()
    rgb_frame1 = frame1[:, :, ::-1]
    
    ret, frame2 = video2.read()
    rgb_frame2 = frame2[:, :, ::-1]
    
    predictions1 = predict(rgb_frame1, model_path="trained_knn_model.clf")
    predictions2 = predict(rgb_frame2, model_path="trained_knn_model.clf")    
    
    for name, (top, right, bottom, left) in predictions1:
        (x,y,w,h)=predictions1[0][1]
        l1 = [(h,x,y,w)]
        nae1.append(name)
    
    for name, (top, right, bottom, left) in predictions2:
        (x,y,w,h)=predictions2[0][1]
        l2 = [(h,x,y,w)]
        nae2.append(name)
    
    track(frame1,l1,nae1,frame2,l2,nae2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret = False

# Release handle to the webcam
video1.release()
video2.release()
cv2.destroyAllWindows()
