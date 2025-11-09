from google.oauth2 import service_account
from googleapiclient.discovery import build
from config import SCOPES

def init_drive_service(service_account_file: str):
    """Initializes the Google Drive service using a service account JSON key."""
    print("1. Initializing the service account...")
    try:
        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)
        print("✅ Initialization successful.")
        return service
    except FileNotFoundError:
        print(f"❌ Error: Key file not found at: {service_account_file}")
        return None
    except Exception as e:
        print(f"❌ Authorization error: Check the key and SCOPES. {e}")
        return None

def list_files_in_folder(service, folder_id: str):
    """Retrieves and lists files (not folders) in the specified folder."""
    print(f"\n2. Attempting to read files in folder ID: {folder_id}...")
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    try:
        response = service.files().list(
            q=query,
            fields='files(id, name, mimeType)',
            pageSize=10
        ).execute()
        files = response.get('files', [])
        if files:
            print(f"✅ Success! Found {len(files)} files (or more).")
            file_data_list = []
            for f in files:
                file_data_list.append({
                    "Имя Файла": f.get('name', 'N/A'),
                    "ID Файла": f.get('id', 'N/A'),
                    "MIME Тип": f.get('mimeType', 'N/A')
                })
            print('Received list of documents: ', file_data_list)
            return file_data_list
        else:
            print("⚠️ Successful connection, but no files found. Please check:")
    except Exception as e:
        print(f"Error requesting file list: {e}")

def download_drive_file_content(drive_service, file_id: str, file_name: str) -> bytes | None:
    """Downloads the content of a file from Google Drive."""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        content = request.execute()
        return content
    except Exception as e:
        print(f"❌ Error downloading file {file_name} ({file_id}): {e}")
        return None

def get_drive_web_link(service, file_id: str) -> str | None:
    """Returns the webViewLink for a Google Drive file by its ID."""
    try:
        file_metadata = service.files().get(
            fileId=file_id,
            fields='webViewLink'
        ).execute()
        return file_metadata.get('webViewLink')
    except Exception as e:
        print(f"❌ Error getting link for file ID {file_id}: {e}")
        return None
