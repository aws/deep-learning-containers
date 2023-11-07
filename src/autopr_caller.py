import json
import boto3
import os
import copy

AUTOPR_PROD_QUEUE = "autopr-prod-queue"
S3_BUCKET = "pr-creation-data-helper"


def get_tag_set_dictionary_from_response(tag_sets):
    tag_set_dict = {}
    for tag_set in tag_sets:
        tag_set_dict[tag_set['Key']] = tag_set['Value']
    return tag_set_dict


def generate_edited_files_data(image_list, bucket=S3_BUCKET, folder="temp"):
    edited_files_data = []
    session = boto3.Session()

    #Then use the session to get the resource
    s3_resource = session.resource('s3')
    s3_client = s3_resource.meta.client

    my_bucket = s3_resource.Bucket(S3_BUCKET)

    for s3_object in my_bucket.objects.filter(Prefix=folder):
        response = s3_client.get_object_tagging(
            Bucket=bucket,
            Key=s3_object.key,
        )
        tag_set_dict = get_tag_set_dictionary_from_response(response['TagSet'])
        upload_path = tag_set_dict.get("upload_path","")
        truncated_upload_path = upload_path.split("deep-learning-containers/")[-1]
        image_uri_corresponding_to_the_file = tag_set_dict.get("image_uri", "")
        if image_uri_corresponding_to_the_file not in image_list:
            continue
        if truncated_upload_path:
            edited_files_data.append({
                "s3_bucket":bucket,
                "s3_filename": s3_object.key,
                "github_filepath": truncated_upload_path,
            })
    
    return edited_files_data


def main():
    image_list = os.getenv("DLC_IMAGES","").split()
    print(image_list)
    edited_files_data = generate_edited_files_data(image_list)
    print(edited_files_data)

if __name__ == "__main__":
    main()
