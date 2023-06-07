from azure.storage.blob import BlobServiceClient
import pickle
import json


# pip install azure-storage-blob
# Once you have created your BlobStorage in Azure - go to the "access keys" tab  in your container to get this information
storage_account_key = ""
storage_account_name = ""
connection_string = ""
container_name = ""


# Create a json object to be shared via blob storage:
# Load the pickled SARIMA model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions for 31 days using the reopened model
pred_uc = model.get_forecast(steps=31)
sales_pred = pred_uc.predicted_mean.to_frame(name='Predicted Sales').values.tolist()
sales_pred_truncated = [round(value[0], 2) for value in sales_pred]
pred_ci = pred_uc.conf_int().values.tolist()
pred_ci_truncated = [round(value[0], 2) for value in pred_ci]

# Create a dictionary to hold the predictions and confidence intervals
data = {
    'sales_predictions': sales_pred_truncated,
    'confidence_intervals': pred_ci_truncated
}

# Convert the dictionary to a JSON object
json_data = json.dumps(data)

# Print or use the JSON object as needed
# print(json_data)

# Upload the JSON string to Blob storage
blob_name = "output.json"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
container_client.upload_blob(name=blob_name, data=json_data, overwrite=True)

# Print the blob URL for future reference
blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
print("Blob URL:", blob_url)











# Upload file from folder to Azure blob storage
# def uploadToBlobStorage(file_path,file_name):
#     blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#     blob_client = blob_service_client.get_blob_client(container = container_name, blob=file_name)
    
#     with open(file_path,'rb') as data:
#         blob_client.upload_blob(data)
#     print(f"Uploaded{file_name}.")
# uploadToBlobStorage('C:/')