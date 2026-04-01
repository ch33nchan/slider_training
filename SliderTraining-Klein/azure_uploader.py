#!/usr/bin/env python3
import sys
import os
import shutil
import io
import uuid
import mimetypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from PIL import Image
from loguru import logger

from utils.storage_utils import Storage

def upload_single_file_to_storage(file_path: str, prefix: str = "general") -> str:
    storage = Storage()
    
    is_temp_zip = False
    
    # Check if the file path is a directory
    if os.path.isdir(file_path):
        logger.info(f"Directory detected, zipping: {file_path}")
        zip_path = shutil.make_archive(file_path, 'zip', file_path)
        file_to_upload = zip_path
        original_name = os.path.basename(file_path) + ".zip"
        is_temp_zip = True
    else:
        file_to_upload = file_path
        original_name = os.path.basename(file_path)
        
    # Guess mime type to set content settings accurately
    mime_type, _ = mimetypes.guess_type(file_to_upload)
    if mime_type is None:
        mime_type = "application/octet-stream"
        
    format_ext = original_name.split('.')[-1] if '.' in original_name else "bin"
    
    with open(file_to_upload, 'rb') as f:
        file_bytes = io.BytesIO(f.read())
        
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"{prefix}/{unique_id}_{original_name}"
    
    logger.info(f"Uploading {file_to_upload} as {blob_name}...")
    
    try:
        # We reuse the same bucket that is known to work from lighting_transfer 
        # or use "user-uploaded-files" from storage_utils. However, "frameo-tools/general_uploads" 
        # could also work if frameo-tools is the container. Let's use frameo-tools/general_uploads.
        urls = storage.bulk_file_upload(
            files=[file_bytes],
            bucket="frameo-tools/general_uploads",
            format=format_ext,
            content_type=mime_type.split('/')[0] if '/' in mime_type else "application",
            fileNames=[blob_name]
        )
        
        if urls and len(urls) > 0:
            return urls[0]
        else:
            raise Exception("Upload returned no URL")
    finally:
        # Clean up the temporary zip if we created one
        if is_temp_zip and os.path.exists(file_to_upload):
            os.remove(file_to_upload)

def upload_and_get_url(file, prefix="single"):
    """Upload any file or directory to Azure and return the URL"""
    if file is None:
        return "No file uploaded"
    
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        logger.info(f"Initiating upload for: {file_path} with prefix: {prefix}")
        url = upload_single_file_to_storage(file_path, prefix=prefix)
        logger.success(f"Upload successful: {url}")
        return url
        
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

def upload_batch(files, prefix="batch"):
    """Upload multiple files/directories and return URLs"""
    if not files:
        return "No files uploaded"
    
    results = []
    for idx, file in enumerate(files):
        try:
            file_path = file.name if hasattr(file, 'name') else str(file)
            url = upload_single_file_to_storage(file_path, prefix=prefix)
            file_name = os.path.basename(file_path)
            results.append(f"{idx+1}. {file_name}: {url}")
            logger.success(f"Uploaded {idx+1}/{len(files)}: {url}")
        except Exception as e:
            file_name_err = os.path.basename(file.name if hasattr(file, 'name') else str(file))
            results.append(f"{idx+1}. ERROR on {file_name_err}: {str(e)}")
            logger.error(f"Failed to upload file {idx+1}: {e}")
    
    return "\n".join(results)


# Create Gradio interface
with gr.Blocks(title="Azure Universal Uploader") as app:
    gr.Markdown("# 🔗 Azure Blob Storage Universal Uploader")
    gr.Markdown("Upload **ANY** file, zip, folder, or images and get Azure CDN URLs instantly")
    
    with gr.Tabs():
        with gr.TabItem("Single File / Zip Upload"):
            with gr.Row():
                with gr.Column():
                    single_file = gr.File(label="Upload Any File or Zip")
                    single_prefix = gr.Textbox(
                        value="single",
                        label="Storage Prefix",
                        placeholder="e.g., single, test, demo"
                    )
                    single_btn = gr.Button("Upload & Get URL", variant="primary")
                
                with gr.Column():
                    single_url = gr.Textbox(
                        label="Azure CDN URL",
                        placeholder="URL will appear here...",
                        lines=3
                    )
            
            single_btn.click(
                upload_and_get_url,
                inputs=[single_file, single_prefix],
                outputs=[single_url]
            )
        
        with gr.TabItem("Batch Files Upload"):
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(
                        file_count="multiple",
                        label="Upload Multiple Files"
                    )
                    batch_prefix = gr.Textbox(
                        value="batch",
                        label="Storage Prefix",
                        placeholder="e.g., dataset, assets"
                    )
                    batch_btn = gr.Button("Upload All & Get URLs", variant="primary")
                
                with gr.Column():
                    batch_urls = gr.Textbox(
                        label="Azure CDN URLs",
                        placeholder="URLs will appear here...",
                        lines=15
                    )
            
            batch_btn.click(
                upload_batch,
                inputs=[batch_files, batch_prefix],
                outputs=batch_urls
            )
            
        with gr.TabItem("Folder Contents Upload"):
            with gr.Row():
                with gr.Column():
                    folder_files = gr.File(
                        file_count="directory",
                        label="Upload a Folder (Uploads all individual files inside)"
                    )
                    folder_prefix = gr.Textbox(
                        value="folder",
                        label="Storage Prefix",
                        placeholder="e.g., my_folder"
                    )
                    folder_btn = gr.Button("Upload Folder Contents", variant="primary")
                
                with gr.Column():
                    folder_urls = gr.Textbox(
                        label="Azure CDN URLs",
                        placeholder="URLs will appear here...",
                        lines=15
                    )
            
            folder_btn.click(
                upload_batch,
                inputs=[folder_files, folder_prefix],
                outputs=folder_urls
            )
    
    gr.Markdown("""
    ### 📝 Notes:
    - Files are uploaded to an Azure Blob Storage Bucket.
    - URLs are served via CDN for fast access.
    - Supported formats: **ANY** (Images, Videos, PDFs, ZIPs, raw data...)
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7878)
