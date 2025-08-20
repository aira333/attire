import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import time
import threading
from queue import Queue
from datetime import datetime

class KYCFaceMatching:
    def __init__(self):
        try:
            self.detector = cv2.FaceDetectorYN.create(
                "face_detection_yunet_2023mar.onnx", "", (320, 320)
            )
            self.detection_method = "yunet"
        except:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detection_method = "haar"
        
        # LBPH face recognition
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Camera setup
        self.cap = None
        self.is_capturing = False
        self.snapshot_queue = Queue()
        
    def _detect_face(self, image):
        """detect face using Haar cascade with parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_params = [
            (1.1, 3, (30, 30)),  # More sensitive
            (1.05, 3, (20, 20)), # Even more sensitive
            (1.3, 5, (50, 50)),  # Less sensitive but more accurate
        ]
        
        for scale, neighbors, min_size in face_params:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale, 
                minNeighbors=neighbors, 
                minSize=min_size
            )
            if len(faces) > 0:
                # Return largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                return largest
        
    def test_camera_and_detection(self): 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return False
            
        print("Camera opened. Press 'q' to quit, 's' to save test image")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # try face detection
                try:
                    face_coords = self._detect_face(frame)
                except:
                    face_coords = None
                
                # draw rectangle if face detected
                if face_coords is not None:
                    x, y, w, h = face_coords
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "FACE DETECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # show frame
                cv2.imshow('Face Detection Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('camera_test.jpg', frame)
                    print("📸 Image saved as camera_test.jpg")
        
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return True
    
    def load_kyc_photo(self, image_path: str) -> bool:
        """load imageKYC and train recognizer"""
        image = cv2.imread(image_path)
        if image is None:
            return False
            
        # detect face
        face_coords = self._detect_face(image)
        if face_coords is None:
            return False
        
        # extract face region
        x, y, w, h = face_coords
        face_img = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # train LBPH recognizer with single KYC face
        self.recognizer.train([face_resized], np.array([0]))
        self.kyc_loaded = True
        return True
    
    def start_camera(self) -> bool:
        """start live camera capture"""
        if not self.kyc_loaded:
            return False
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
            
        self.is_capturing = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return True
    
    def _capture_loop(self):
        """camera capture loop"""
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                self.snapshot_queue.put(frame.copy())
            time.sleep(0.1)
    
    def verify_live_face(self) -> Dict:
        """verify current live face against imageKYC"""
        if self.snapshot_queue.empty():
            return {"match": False, "confidence": 0, "error": "No camera feed"}
        
        frame = self.snapshot_queue.get()
        print(f"Frame size: {frame.shape}")
        
        # detect face
        face_coords = self._detect_face(frame)
        if face_coords is None:
            return {"match": False, "confidence": 0, "error": "No face detected"}
        
        print(f"Face detected: {face_coords}")
        
        # extract face region  
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # predict with LBPH
        label, confidence = self.recognizer.predict(face_resized)
        
        # convert confidence to match percentage (lower distance = higher confidence)
        match_confidence = max(0, 100 - confidence)
        is_match = confidence < 50  # threshold for match
        
        return {
            "match": is_match,
            "confidence": match_confidence,
            "lbph_distance": confidence
        }
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()

def main():
    kyc = KYCFaceMatching()
    
    print("KYC Face Matching with LBPH")
    print("Detection method:", kyc.detection_method)
    
    # test camera and face detection
    print("\n testing camera and face detection...")
    kyc.test_camera_and_detection()
    
    # load KYC photo
    if not kyc.load_kyc_photo("candidate_kyc_photo.jpg"):
        print("Failed to load KYC photo")
        return
    print("KYC photo loaded")
    
    # start camera
    if not kyc.start_camera():
        print("Failed to start camera")
        return
    print("Camera started")
    
    # live verification loop
    try:
        for i in range(10):
            time.sleep(2)
            result = kyc.verify_live_face()
            
            if "error" in result:
                print(f"Check {i+1}: {result['error']}")
            else:
                status = "✅ MATCH" if result["match"] else "❌ NO MATCH"
                print(f"Check {i+1}: {status} | Confidence: {result['confidence']:.1f}%")
    
    except KeyboardInterrupt:
        pass
    
    kyc.stop_camera()
    print("Done")

if __name__ == "__main__":
    main()