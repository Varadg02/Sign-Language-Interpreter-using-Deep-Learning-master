import sqlite3
import os

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    conn = sqlite3.connect("gesture_db.db")
    create_table_cmd = """
    CREATE TABLE IF NOT EXISTS gesture (
        g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        g_name TEXT NOT NULL
    )
    """
    conn.execute(create_table_cmd)
    conn.commit()
    conn.close()

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = f"INSERT INTO gesture (g_id, g_name) VALUES ({g_id}, '{g_name}')"
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        print(f"Gesture ID {g_id} already exists.")
    conn.commit()
    conn.close()

init_create_folder_database()

# Store Alphabets and Numbers in Database
gestures = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
for i, gesture in enumerate(gestures):
    store_in_db(i, gesture)

print("Gestures stored in database successfully!")

import os
import shutil

# Define the gestures directory
gestures_dir = "gestures"

# Ensure the gestures directory exists
if not os.path.exists(gestures_dir):
    os.mkdir(gestures_dir)

# Define the classes (A-Z, 0-9)
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Create subfolders for each gesture
for gesture in classes:
    gesture_folder = os.path.join(gestures_dir, gesture)
    if not os.path.exists(gesture_folder):
        os.mkdir(gesture_folder)

# Move images into the correct folders (Make sure all images are in a single source folder)
source_folder = "gestures"  # Change this to where your images are stored

for image in os.listdir(source_folder):
    if image.endswith(".jpg"):  # Ensure only images are processed
        label = image[0].upper()  # Extract the first character of the filename as the label
        if label in classes:
            dest_folder = os.path.join(gestures_dir, label)
            shutil.move(os.path.join(source_folder, image), os.path.join(dest_folder, image))

print("Images organized successfully!")
