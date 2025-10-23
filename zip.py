import zipfile
from io import BytesIO

def zip_files_from_filesystem(filenames: list[str]) -> bytes:
    # Create an in-memory bytes buffer.
    zip_buffer = BytesIO()
    
    # Create a ZIP file in memory
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
        for filename in filenames:
            # Archives file from filesystem
            zip.write(filename)
    # Set file pointer to the begining of the zip archive and return the bytes value.
    zip_buffer.seek(0)
    return zip_buffer.getvalue()




