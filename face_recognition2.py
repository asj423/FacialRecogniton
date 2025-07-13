#import these 5 libraries
import numpy as np  # Library for numerical operations
import os           # Library for interacting with the operating system
import cv2          # OpenCV library for computer vision tasks
import sqlite3      # Library used for managing and creating databases
import datetime     # Library for current date and time

# Get the current date and time
current_datetime = datetime.datetime.now()

# Creates a function to find the distance using Pythagoras
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

# Training the data
def knn(train, test, k=5):

    dist = []

    # Converts data into vector form
    for i in range(train.shape[0]):

        ix = train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])

    # Sorted in order of distance
    ds = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(ds)[:, -1]

    output = np.unique(labels, return_counts=True)

    index = np.argmax(output[1])
    return output[0][index]

# Loads pre-trained face cascade classifier
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Captures live video feed from the default camera
LiveFeed = cv2.VideoCapture(0)

# Path to the directory containing face datasets
dataset_path = "./face_dataset/"

# Lists to store face data and corresponding labels
face_data = []
labels = []

# Initializing variables
class_id = 0
names = {}

# Finds files ending in '.npy' and removes the '.npy' extension
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print(names)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Creates an array of ones based on the shape and size of the face
        # and multiplies it by the class_id
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Combines face data and labels into single arrays
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

# Prints the shapes of face labels and dataset
print(face_labels.shape)
print(face_dataset.shape)

# Combines face dataset and labels into a single training set
trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Font for displaying text on images
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to create a database and connect to it
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to database: {db_file}")
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to create tables
def create_tables(conn):
    try:
        cursor = conn.cursor()
        # Create students table
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                            person_id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            year_group INTEGER
                        )''')
        # Create emergency_contacts_id table
        cursor.execute('''CREATE TABLE IF NOT EXISTS contacts (
                            emergency_contacts_id INTEGER PRIMARY KEY,
                            relationship_1 TEXT NOT NULL,
                            relationship_2 TEXT NOT NULL,
                            contact_nuber_1 INTEGER,
                            contact_number_2 INTEGER,
                            primary_address TEXT NOT NULL
                        )''')
        # Create table to handle student-contacts relationships
        cursor.execute('''CREATE TABLE IF NOT EXISTS combined (
                            combined_id INTEGER PRIMARY KEY,
                            person_id INTEGER,
                            emergency_contacts_id INTEGER,
                            FOREIGN KEY (person_id) REFERENCES students (person_id),
                            FOREIGN KEY (emergency_contacts_id) REFERENCES contacts (emergency_contacts_id)
                        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS register (
                            sign_in_id INTEGER PRIMARY KEY,
                            combined_id INTEGER,
                            sign_in BOOLEAN,
                            time DATETIME,
                            FOREIGN KEY (combined_id) REFERENCES combined (combined_id)
                        )''')
        print("Tables created successfully")
    except sqlite3.Error as e:
        print(e)

# Function to insert data into tables
def insert_data(conn, table, data):
    try:
        cursor = conn.cursor()
        cursor.executemany(f"INSERT INTO {table} VALUES ({','.join(['?']*len(data[0]))})", data)
        conn.commit()
        print("Data inserted successfully")
    except sqlite3.Error as e:
        print(e)

# Function to fetch and display data
def fetch_data(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
    except sqlite3.Error as e:
        print(e)

# Main loop for capturing and processing frames from the live video feed
while True:
    _, img = LiveFeed.read()

    # Converts image to grayscale for face detection
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces in the grayscale image
    faces = cascade_face.detectMultiScale(grayImage, 1.1, 4)

    # Sorts detected faces based on area (largest first)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    for face in faces[:1]:

        if __name__ == '__main__':
            database = "school_registration.db"
            conn = create_connection(database)
            if conn is not None:
                create_tables(conn)

                # Fetch and display data from students table
                print("\nStudents:")
                fetch_data(conn, "SELECT * FROM students")

                # Fetch and display data from courses table
                print("\nContacts:")
                fetch_data(conn, "SELECT * FROM contacts")

                print("\nSign_in:")
                fetch_data(conn, "SELECT * FROM register")

            conn.close()

        x, y, w, h = face

        # Extracts face region with an offset for better recognition
        offset = 5
        face_offset = img[y - offset:y + h + offset, x - offset:x + h + offset]
        face_selection = cv2.resize(face_offset, (100, 100))

        # Uses KNN algorithm to predict the label for the face
        out = knn(trainset, face_selection.flatten())

        # Get the person's ID from the names dictionary
        person_name = str(names[int(out)].title())

    # Update the 'register' table to mark the person as signed in
    database = "school_registration.db"
    conn = create_connection(database)
    cursor = conn.cursor()
    cursor.execute("SELECT person_id FROM students WHERE name = ?", (person_name,))
    student_name_id = cursor.fetchone()[0]
    print(student_name_id)
    combined_person_id = cursor.execute("SELECT combined_id FROM combined WHERE person_id = ?", (student_name_id,))
    combined_person_id = combined_person_id.fetchone()[0]
    print(combined_person_id)
    print("hello 2")
    cursor.execute('SELECT COUNT(*) FROM register')
    length = cursor.fetchone()[0]
    query = "INSERT INTO register (sign_in_id, combined_id, sign_in, time) VALUES (?, ?, ?, ?)"
    values = (length, combined_person_id, 1, datetime.datetime.now())
    cursor.execute(query, values)
    conn.commit()

    # Draws rectangle around detected faces and displays name on top-left corner
    for (x, y, w, h) in faces:
        cv2.putText(img, names[int(out)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Displays the processed image
    cv2.imshow('img', img)

    # Waits for 'Esc' key press to exit the loop
    U = cv2.waitKey(30) & 0xff
    if U == 27:
        break
