from deepface import DeepFace

# Find the closest match from the database
result = DeepFace.find(img_path="img1.jpg", db_path="my_db")

# Print the result
print(result)