import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import pyttsx3

class SignLanguageToSpeech:
    """
    Real-time Sign Language to Speech Conversion System
    Features:
    - Real-time hand sign recognition
    - Character-by-character sentence building
    - Text-to-speech conversion
    - Clean GUI interface
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language To Text Conversion")
        self.root.configure(bg='white')
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 1.0)  # Volume level
        
        # Load the trained model
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        
        # Labels dictionary for predictions
        self.labels_dict = {}
        for i in range(26):
            self.labels_dict[i] = chr(65 + i)  # A-Z
        for i in range(10):
            self.labels_dict[26 + i] = str(i)  # 0-9
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # State variables
        self.current_sentence = ""
        self.current_character = ""
        self.last_character = ""
        self.character_count = 0
        self.stable_frames = 0
        self.stability_threshold = 10  # Number of frames to consider stable
        
        # Setup GUI
        self.setup_gui()
        
        # Start video update
        self.is_running = True
        self.update_frame()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Sign Language To Text Conversion",
            font=("Courier New", 24, "bold"),
            bg='white',
            fg='black'
        )
        title_label.pack(pady=(0, 20))
        
        # Content frame (horizontal layout)
        content_frame = tk.Frame(main_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Video feed
        video_frame = tk.Frame(content_frame, bg='black', relief=tk.SUNKEN, borderwidth=2)
        video_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()
        
        # Right side - Hand landmark visualization
        landmark_frame = tk.Frame(content_frame, bg='white', relief=tk.SUNKEN, borderwidth=2)
        landmark_frame.pack(side=tk.LEFT)
        
        self.landmark_canvas = tk.Canvas(
            landmark_frame,
            width=400,
            height=400,
            bg='white',
            highlightthickness=0
        )
        self.landmark_canvas.pack()
        
        # Character display
        char_frame = tk.Frame(main_frame, bg='white')
        char_frame.pack(pady=20)
        
        tk.Label(
            char_frame,
            text="Character :",
            font=("Arial", 18, "bold"),
            bg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        self.char_label = tk.Label(
            char_frame,
            text="",
            font=("Arial", 18, "bold"),
            bg='white',
            fg='black'
        )
        self.char_label.pack(side=tk.LEFT, padx=5)
        
        # Sentence display and controls
        sentence_frame = tk.Frame(main_frame, bg='white')
        sentence_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(
            sentence_frame,
            text="Sentence :",
            font=("Arial", 14, "bold"),
            bg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        self.sentence_entry = tk.Entry(
            sentence_frame,
            font=("Arial", 14),
            width=40,
            state='readonly',
            readonlybackground='white',
            fg='black'
        )
        self.sentence_entry.pack(side=tk.LEFT, padx=10)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(pady=10)
        
        # Clear button
        self.clear_btn = tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 12, "bold"),
            bg='#d3d3d3',
            fg='black',
            padx=20,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2,
            command=self.clear_sentence,
            cursor='hand2'
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Speak button
        self.speak_btn = tk.Button(
            button_frame,
            text="Speak",
            font=("Arial", 12, "bold"),
            bg='#d3d3d3',
            fg='black',
            padx=20,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2,
            command=self.speak_sentence,
            cursor='hand2'
        )
        self.speak_btn.pack(side=tk.LEFT, padx=10)
        
        # Space button (to add space between words)
        self.space_btn = tk.Button(
            button_frame,
            text="Space",
            font=("Arial", 12, "bold"),
            bg='#d3d3d3',
            fg='black',
            padx=20,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2,
            command=self.add_space,
            cursor='hand2'
        )
        self.space_btn.pack(side=tk.LEFT, padx=10)
        
        # Backspace button
        self.backspace_btn = tk.Button(
            button_frame,
            text="Backspace",
            font=("Arial", 12, "bold"),
            bg='#d3d3d3',
            fg='black',
            padx=20,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2,
            command=self.backspace,
            cursor='hand2'
        )
        self.backspace_btn.pack(side=tk.LEFT, padx=10)
        
    def draw_hand_landmarks(self, hand_landmarks):
        """Draw hand landmarks on the canvas"""
        self.landmark_canvas.delete("all")
        
        if hand_landmarks is None:
            return
        
        # Canvas dimensions
        canvas_width = 400
        canvas_height = 400
        
        # Draw connections
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = hand_landmarks.landmark[start_idx]
            end_point = hand_landmarks.landmark[end_idx]
            
            start_x = int(start_point.x * canvas_width)
            start_y = int(start_point.y * canvas_height)
            end_x = int(end_point.x * canvas_width)
            end_y = int(end_point.y * canvas_height)
            
            self.landmark_canvas.create_line(
                start_x, start_y, end_x, end_y,
                fill='#00FF00',
                width=2
            )
        
        # Draw landmarks
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * canvas_width)
            y = int(landmark.y * canvas_height)
            
            self.landmark_canvas.create_oval(
                x-5, y-5, x+5, y+5,
                fill='#00FF00',
                outline='#00FF00'
            )
    
    def update_frame(self):
        """Update video frame and perform prediction"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        
        H, W, _ = frame.shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        predicted_character = ""
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on video frame
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract features for prediction
            data_aux = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            
            # Get bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            
            # Make prediction
            try:
                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]
                
                # Draw bounding box and prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )
                
                # Update character with stability check
                self.update_character(predicted_character)
                
            except Exception as e:
                print(f"Prediction error: {e}")
            
            # Draw hand landmarks on canvas
            self.draw_hand_landmarks(hand_landmarks)
        else:
            # Clear canvas if no hand detected
            self.landmark_canvas.delete("all")
            self.stable_frames = 0
        
        # Convert frame for tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Schedule next update
        self.root.after(10, self.update_frame)
    
    def update_character(self, character):
        """Update current character with stability check"""
        if character == self.last_character:
            self.stable_frames += 1
        else:
            self.stable_frames = 1
            self.last_character = character
        
        # Only update if character is stable
        if self.stable_frames >= self.stability_threshold:
            if self.current_character != character:
                self.current_character = character
                self.char_label.config(text=character)
                
                # Auto-add character to sentence after being stable
                if self.stable_frames == self.stability_threshold:
                    self.add_to_sentence(character)
    
    def add_to_sentence(self, character):
        """Add character to sentence"""
        self.current_sentence += character
        self.update_sentence_display()
        self.stable_frames = 0  # Reset after adding
    
    def update_sentence_display(self):
        """Update sentence entry display"""
        self.sentence_entry.config(state='normal')
        self.sentence_entry.delete(0, tk.END)
        self.sentence_entry.insert(0, self.current_sentence)
        self.sentence_entry.config(state='readonly')
    
    def add_space(self):
        """Add space to sentence"""
        self.current_sentence += " "
        self.update_sentence_display()
    
    def backspace(self):
        """Remove last character from sentence"""
        if self.current_sentence:
            self.current_sentence = self.current_sentence[:-1]
            self.update_sentence_display()
    
    def clear_sentence(self):
        """Clear the sentence"""
        self.current_sentence = ""
        self.current_character = ""
        self.char_label.config(text="")
        self.update_sentence_display()
    
    def speak_sentence(self):
        """Convert sentence to speech"""
        if self.current_sentence.strip():
            # Run TTS in separate thread to avoid blocking GUI
            threading.Thread(
                target=self._speak_thread,
                args=(self.current_sentence,),
                daemon=True
            ).start()
    
    def _speak_thread(self, text):
        """Thread function for text-to-speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SignLanguageToSpeech(root)
    
    # Handle window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
