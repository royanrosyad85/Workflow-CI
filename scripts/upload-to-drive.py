from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

SERVICE_ACCOUNT_FILE = 'service_account.json'
FOLDER_ID = "18cPLEVjDOW8rb0veQVZh8FXojPqFexmi"
ARTIFACTS_DIR = "../model_output"  # Changed path to look in the parent directory

SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# Print working directory and listing for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Checking if {ARTIFACTS_DIR} exists: {os.path.exists(ARTIFACTS_DIR)}")
if os.path.exists(ARTIFACTS_DIR):
    print(f"Contents of {ARTIFACTS_DIR}:")
    print(os.listdir(ARTIFACTS_DIR))

for root, dirs, files in os.walk(ARTIFACTS_DIR):
    for filename in files:
        file_path = os.path.join(root, filename)
        file_metadata = {'name': filename, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()
        print(f"Uploaded {filename} with ID: {file.get('id')}")