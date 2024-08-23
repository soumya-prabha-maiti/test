import datetime
import json
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build


def upload_file_to_gdrive(
    file_path, parent_folder_ids: list[str], creds_stringified: str | None = None
):
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    if not creds_stringified:
        print("Attempting to use google drive creds from environment variable")
        creds_stringified = os.getenv("CREDS")
    creds_dict = json.loads(creds_stringified)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)
    file_metadata = {"name": file_path, "parents": parent_folder_ids}
    file = service.files().create(body=file_metadata, media_body=file_path).execute()
    print("File uploaded, file id: ", file.get("id"))


def upload_file_to_gdrive_sanity_check(
    parent_folder_ids: list[str], creds_stringified: str | None = None
):
    try:
        curr_time_utc = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"gdrive_upload_test_{curr_time_utc}_UTC.txt"
        with open(file_name, "w") as f:
            f.write(f"gdrive_upload_test_{curr_time_utc}_UTC")
        upload_file_to_gdrive(file_name, parent_folder_ids, creds_stringified)
        os.remove(file_name)
    except Exception as e:
        if os.path.exists(file_name):
            os.remove(file_name)
        raise e
