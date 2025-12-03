"""Azure Blob Storage manager for notebook and general files"""
import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reduce Azure SDK logging verbosity
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

class AzureBlobManager:
    """Manages files in Azure Blob Storage with separate containers for notebooks and uploads"""

    def __init__(self):
        """Initialize Azure Blob Storage client"""
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")

        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Two containers: one for notebooks, one for general uploads
        self.notebooks_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME", "notebooks")
        self.uploads_container_name = "uploads"

        # Get container clients
        self.notebooks_container = self.blob_service_client.get_container_client(self.notebooks_container_name)
        self.uploads_container = self.blob_service_client.get_container_client(self.uploads_container_name)

        # Ensure containers exist
        self._ensure_container_exists(self.notebooks_container)
        self._ensure_container_exists(self.uploads_container)

    def _ensure_container_exists(self, container_client: ContainerClient):
        """Ensure a container exists, create if not"""
        try:
            container_client.get_container_properties()
        except ResourceNotFoundError:
            container_client.create_container()

    # ==================== Notebook Management ====================

    def upload_notebook(self, filename: str, content: bytes) -> Dict:
        """
        Upload a notebook file to Azure Blob Storage (notebooks container)
        Only accepts .ipynb files with valid Jupyter format

        Args:
            filename: Name of the notebook file
            content: File content as bytes

        Returns:
            Dict with upload status and metadata
        """
        try:
            # Validate file extension
            if not filename.endswith('.ipynb'):
                raise ValueError("Only .ipynb files are allowed")

            # Validate notebook JSON structure
            try:
                nb_data = json.loads(content.decode('utf-8'))
                if "cells" not in nb_data or "metadata" not in nb_data:
                    raise ValueError("Invalid Jupyter notebook format")
            except json.JSONDecodeError:
                raise ValueError("File is not valid JSON")

            # Upload to notebooks container
            blob_client = self.notebooks_container.get_blob_client(filename)
            blob_client.upload_blob(content, overwrite=True)

            # Get blob properties
            properties = blob_client.get_blob_properties()

            return {
                "status": "success",
                "filename": filename,
                "size": properties.size,
                "uploaded_at": properties.last_modified.isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to upload notebook: {str(e)}")

    def download_notebook(self, filename: str) -> bytes:
        """
        Download a notebook file from Azure Blob Storage

        Args:
            filename: Name of the notebook file

        Returns:
            File content as bytes
        """
        try:
            blob_client = self.notebooks_container.get_blob_client(filename)
            download_stream = blob_client.download_blob()
            return download_stream.readall()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Notebook not found: {filename}")
        except Exception as e:
            raise Exception(f"Failed to download notebook: {str(e)}")

    def list_notebooks(self) -> List[Dict]:
        """
        List all notebook files in Azure Blob Storage

        Returns:
            List of notebook metadata dictionaries
        """
        try:
            notebooks = []
            blob_list = self.notebooks_container.list_blobs()

            for blob in blob_list:
                if blob.name.endswith('.ipynb'):
                    notebooks.append({
                        "filename": blob.name,
                        "size": blob.size,
                        "modified_at": blob.last_modified.isoformat()
                    })

            # Sort by modified time, most recent first
            notebooks.sort(key=lambda x: x["modified_at"], reverse=True)

            return notebooks
        except Exception as e:
            raise Exception(f"Failed to list notebooks: {str(e)}")

    def delete_notebook(self, filename: str) -> Dict:
        """
        Delete a notebook file from Azure Blob Storage

        Args:
            filename: Name of the notebook file

        Returns:
            Dict with deletion status
        """
        try:
            blob_client = self.notebooks_container.get_blob_client(filename)
            blob_client.delete_blob()

            return {
                "status": "success",
                "message": f"Deleted {filename}"
            }
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Notebook not found: {filename}")
        except Exception as e:
            raise Exception(f"Failed to delete notebook: {str(e)}")

    def get_notebook_metadata(self, filename: str) -> Dict:
        """
        Get metadata for a specific notebook file

        Args:
            filename: Name of the notebook file

        Returns:
            Dict with notebook metadata
        """
        try:
            blob_client = self.notebooks_container.get_blob_client(filename)
            properties = blob_client.get_blob_properties()

            return {
                "filename": filename,
                "size": properties.size,
                "modified_at": properties.last_modified.isoformat(),
                "content_type": properties.content_settings.content_type
            }
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Notebook not found: {filename}")
        except Exception as e:
            raise Exception(f"Failed to get notebook metadata: {str(e)}")

    def notebook_exists(self, filename: str) -> bool:
        """
        Check if a notebook file exists in Azure Blob Storage

        Args:
            filename: Name of the notebook file

        Returns:
            True if notebook exists, False otherwise
        """
        try:
            blob_client = self.notebooks_container.get_blob_client(filename)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception:
            return False

    # ==================== General File Management ====================

    def upload_file(self, filename: str, content: bytes) -> Dict:
        """
        Upload any file to Azure Blob Storage (uploads container)
        No file type restrictions

        Args:
            filename: Name of the file
            content: File content as bytes

        Returns:
            Dict with upload status and metadata
        """
        try:
            # Upload to uploads container
            blob_client = self.uploads_container.get_blob_client(filename)
            blob_client.upload_blob(content, overwrite=True)

            # Get blob properties
            properties = blob_client.get_blob_properties()

            return {
                "status": "success",
                "filename": filename,
                "size": properties.size,
                "uploaded_at": properties.last_modified.isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    def download_file(self, filename: str) -> bytes:
        """
        Download a file from Azure Blob Storage (uploads container)

        Args:
            filename: Name of the file

        Returns:
            File content as bytes
        """
        try:
            blob_client = self.uploads_container.get_blob_client(filename)
            download_stream = blob_client.download_blob()
            return download_stream.readall()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")

    def list_files(self) -> List[Dict]:
        """
        List all files in Azure Blob Storage (uploads container)

        Returns:
            List of file metadata dictionaries
        """
        try:
            files = []
            blob_list = self.uploads_container.list_blobs()

            for blob in blob_list:
                files.append({
                    "filename": blob.name,
                    "size": blob.size,
                    "modified_at": blob.last_modified.isoformat()
                })

            # Sort by modified time, most recent first
            files.sort(key=lambda x: x["modified_at"], reverse=True)

            return files
        except Exception as e:
            raise Exception(f"Failed to list files: {str(e)}")

    def delete_file(self, filename: str) -> Dict:
        """
        Delete a file from Azure Blob Storage (uploads container)

        Args:
            filename: Name of the file

        Returns:
            Dict with deletion status
        """
        try:
            blob_client = self.uploads_container.get_blob_client(filename)
            blob_client.delete_blob()

            return {
                "status": "success",
                "message": f"Deleted {filename}"
            }
        except ResourceNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        except Exception as e:
            raise Exception(f"Failed to delete file: {str(e)}")

    def file_exists(self, filename: str) -> bool:
        """
        Check if a file exists in Azure Blob Storage (uploads container)

        Args:
            filename: Name of the file

        Returns:
            True if file exists, False otherwise
        """
        try:
            blob_client = self.uploads_container.get_blob_client(filename)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception:
            return False


# Singleton instance
_blob_manager_instance = None

def get_blob_manager() -> AzureBlobManager:
    """Get or create singleton instance of AzureBlobManager"""
    global _blob_manager_instance
    if _blob_manager_instance is None:
        _blob_manager_instance = AzureBlobManager()
    return _blob_manager_instance
