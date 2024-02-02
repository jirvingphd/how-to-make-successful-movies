
# 📚 Using `boto3` - AWS' Python SDK


```python
!pip install -U boto3
```

    Requirement already satisfied: boto3 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (1.20.31)
    Requirement already satisfied: botocore<1.24.0,>=1.23.31 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (from boto3) (1.23.31)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (from boto3) (0.10.0)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (from boto3) (0.5.0)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (from botocore<1.24.0,>=1.23.31->boto3) (2.8.1)
    Requirement already satisfied: urllib3<1.27,>=1.25.4 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (from botocore<1.24.0,>=1.23.31->boto3) (1.25.10)
    Requirement already satisfied: six>=1.5 in /Users/codingdojo/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.24.0,>=1.23.31->boto3) (1.15.0)


### Documentation/Resources
- [AWS' PYTHON SDK](https://aws.amazon.com/sdk-for-python/)

## AWS/`boto3` Requirements/Setup

### AWS credentials stored locally

There are several ways to provide the access ID and secret key to access AWS services. 
- Using the AWS config file.
- Passing credentials as parameters
- Using shared credentials file
- Using environment variables



> Reference: https://www.stackvidhya.com/specify-credentials-in-boto3/

#### Notes:
- I chose to use the AWS config file method.
- If I had stored the credentials in a json file instead, I would use the following code with boto3:
```python
import boto3
session = boto3.Session(
    aws_access_key_id='<your_access_key_id>',
    aws_secret_access_key='<your_secret_access_key>'
)
```

## `boto3` connections/classes

- From what I've seen so far:
    - `boto3` has 3 main classes/types of connections that are used for different things, but have some overlapping functionality.
    - `Sessions`, `clients`, and `resources`

### `Session`s

- Sessions will allow progamattic access to some helpful pieces of info, like what AWS region we are currently using. (e.g. `us-east-2`)
- You can specify credentials when using `boto3`:
    - How-to Article: 
        - https://www.stackvidhya.com/specify-credentials-in-boto3/

```python
## Create a session object to programmatically get region info
session = boto3.session.Session()
session.region_name
```


```python
# !pip install -U boto3
import boto3
```


```python
## Using teacher credentials with read/write access
my_creds = pd.read_csv("/Users/codingdojo/.secret/aws-jirvingphd-credentials.csv")
my_creds.columns
```




    Index(['User name', 'Password', 'Access key ID', 'Secret access key',
           'Console login link'],
          dtype='object')




```python
## Create a session object to programmatically get region info
session = boto3.Session(
    aws_access_key_id=my_creds.loc[0,'Access key ID'],
    aws_secret_access_key=my_creds.loc[0,'Secret access key'],)
session
```




    Session(region_name='us-east-2')




```python
## Check what resources are available
session.get_available_resources()
```




    ['cloudformation',
     'cloudwatch',
     'dynamodb',
     'ec2',
     'glacier',
     'iam',
     'opsworks',
     's3',
     'sns',
     'sqs']




```python
def check_methods_attributes(var, filter_out='_'):
    print('Attributes and Methods:')
    [print(f"- self.{c}") for c in sorted(dir(var)) if not c.startswith(filter_out)];
```


```python
## Can loop up credentials currently in use 
sess_creds = session.get_credentials()
check_methods_attributes(sess_creds)
# print('Attributes and Methods:')
# [print(f"- {c}") for c in sorted(dir(sess_creds)) if not c.startswith('__')];
```

    Attributes and Methods:
    - self.access_key
    - self.get_frozen_credentials
    - self.method
    - self.secret_key
    - self.token


### `resource` vs `client`

- Two kinds
    - boto3.resource:
        - OOP methods/easier to work with
        - Usually resources are what we want to use. (one exception is loading data ***directly*** into pandas)
    - boto3.client:
        - Returns JSON/dictionary results
        - Required to load data directly into a dataframe wihtout saving a file first

#### `boto3.resource`


```python
## import boto3 and create a RESOURCE (not a CLIENT)
import boto3
s3_resource = boto3.resource('s3')
s3_resource
```




    s3.ServiceResource()




```python
check_methods_attributes(s3_resource)
```

    Attributes and Methods:
    - self.Bucket
    - self.BucketAcl
    - self.BucketCors
    - self.BucketLifecycle
    - self.BucketLifecycleConfiguration
    - self.BucketLogging
    - self.BucketNotification
    - self.BucketPolicy
    - self.BucketRequestPayment
    - self.BucketTagging
    - self.BucketVersioning
    - self.BucketWebsite
    - self.MultipartUpload
    - self.MultipartUploadPart
    - self.Object
    - self.ObjectAcl
    - self.ObjectSummary
    - self.ObjectVersion
    - self.buckets
    - self.create_bucket
    - self.get_available_subresources
    - self.meta


#### `boto3.client`


```python
s3_client = boto3.client('s3')
s3_client
```




    <botocore.client.S3 at 0x7f9709f06bd0>




```python
check_methods_attributes(s3_client)
```

    Attributes and Methods:
    - self.abort_multipart_upload
    - self.can_paginate
    - self.complete_multipart_upload
    - self.copy
    - self.copy_object
    - self.create_bucket
    - self.create_multipart_upload
    - self.delete_bucket
    - self.delete_bucket_analytics_configuration
    - self.delete_bucket_cors
    - self.delete_bucket_encryption
    - self.delete_bucket_intelligent_tiering_configuration
    - self.delete_bucket_inventory_configuration
    - self.delete_bucket_lifecycle
    - self.delete_bucket_metrics_configuration
    - self.delete_bucket_ownership_controls
    - self.delete_bucket_policy
    - self.delete_bucket_replication
    - self.delete_bucket_tagging
    - self.delete_bucket_website
    - self.delete_object
    - self.delete_object_tagging
    - self.delete_objects
    - self.delete_public_access_block
    - self.download_file
    - self.download_fileobj
    - self.exceptions
    - self.generate_presigned_post
    - self.generate_presigned_url
    - self.get_bucket_accelerate_configuration
    - self.get_bucket_acl
    - self.get_bucket_analytics_configuration
    - self.get_bucket_cors
    - self.get_bucket_encryption
    - self.get_bucket_intelligent_tiering_configuration
    - self.get_bucket_inventory_configuration
    - self.get_bucket_lifecycle
    - self.get_bucket_lifecycle_configuration
    - self.get_bucket_location
    - self.get_bucket_logging
    - self.get_bucket_metrics_configuration
    - self.get_bucket_notification
    - self.get_bucket_notification_configuration
    - self.get_bucket_ownership_controls
    - self.get_bucket_policy
    - self.get_bucket_policy_status
    - self.get_bucket_replication
    - self.get_bucket_request_payment
    - self.get_bucket_tagging
    - self.get_bucket_versioning
    - self.get_bucket_website
    - self.get_object
    - self.get_object_acl
    - self.get_object_legal_hold
    - self.get_object_lock_configuration
    - self.get_object_retention
    - self.get_object_tagging
    - self.get_object_torrent
    - self.get_paginator
    - self.get_public_access_block
    - self.get_waiter
    - self.head_bucket
    - self.head_object
    - self.list_bucket_analytics_configurations
    - self.list_bucket_intelligent_tiering_configurations
    - self.list_bucket_inventory_configurations
    - self.list_bucket_metrics_configurations
    - self.list_buckets
    - self.list_multipart_uploads
    - self.list_object_versions
    - self.list_objects
    - self.list_objects_v2
    - self.list_parts
    - self.meta
    - self.put_bucket_accelerate_configuration
    - self.put_bucket_acl
    - self.put_bucket_analytics_configuration
    - self.put_bucket_cors
    - self.put_bucket_encryption
    - self.put_bucket_intelligent_tiering_configuration
    - self.put_bucket_inventory_configuration
    - self.put_bucket_lifecycle
    - self.put_bucket_lifecycle_configuration
    - self.put_bucket_logging
    - self.put_bucket_metrics_configuration
    - self.put_bucket_notification
    - self.put_bucket_notification_configuration
    - self.put_bucket_ownership_controls
    - self.put_bucket_policy
    - self.put_bucket_replication
    - self.put_bucket_request_payment
    - self.put_bucket_tagging
    - self.put_bucket_versioning
    - self.put_bucket_website
    - self.put_object
    - self.put_object_acl
    - self.put_object_legal_hold
    - self.put_object_lock_configuration
    - self.put_object_retention
    - self.put_object_tagging
    - self.put_public_access_block
    - self.restore_object
    - self.select_object_content
    - self.upload_file
    - self.upload_fileobj
    - self.upload_part
    - self.upload_part_copy
    - self.waiter_names
    - self.write_get_object_response


##  🪣**Accessing Data in S3 Buckets**

- https://realpython.com/python-boto3-aws-s3/
- https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html

>- Create a `boto3.resource` for "s3"
    


```python
### create a resources
s3_res= boto3.resource('s3')
s3_res
```




    s3.ServiceResource()




```python
print(f"[i] All Current Buckets:")
[print(f"- {b}") for b in s3_res.buckets.all()];
```

    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')


### Creating a New Bucket


```python
## BUCKET TO CREATE
BUCKET_NAME = 'data-enrichment-belt-exam'
```


```python
## Get list of buckets
bucket_list = [b for b in s3_res.buckets.all()]
bucket_list
```




    [s3.Bucket(name='data-enrichment-belt-exam')]




```python
## Testing a single bucket from list
b = bucket_list[0]
b.name
```




    'data-enrichment-belt-exam'




```python
def check_buckets(s3_res,return_names=False):
    print(f"[i] All Current Buckets:")
    bucket_list = [b for b in s3_res.buckets.all()]
    [print(f"- {b}") for b in bucket_list];
    
    if return_names:
        return [b.name for b in bucket_list]
    
```


```python
## Get list of buckets with func
bucket_names = check_buckets(s3_res, return_names=True)
bucket_names
```

    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')





    ['data-enrichment-belt-exam']




```python
check_buckets(s3_res)
```

    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')



```python
# get names of buckets
bucket_names = [b.name for b in bucket_list]
bucket_names
```




    ['data-enrichment-belt-exam']




```python
check_buckets(s3_res)
```

    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')



```python
# bucket_name = 'example-bucket-for-lesson2'
# bucket = s3_resource.create_bucket(Bucket=bucket_name,
#                           CreateBucketConfiguration={
#                               'LocationConstraint': session.region_name})
# bucket
```

### Deleting a Bucket

- Buckets must be completely empty before they can be deleted. 
- The deletion process takes several minutes


```python
def delete_all_objects(s3_resource, bucket_name):
    """Source: https://realpython.com/python-boto3-aws-s3/"""

    res = []
    # get bucket 
    bucket=s3_resource.Bucket(bucket_name)
    
    # get all object versions
    for obj_version in bucket.object_versions.all():
        #saving the object_jet and version id
        res.append({'Key': obj_version.object_key,
                    'VersionId': obj_version.id})
#     print(res)
    
    ## Delete objects
    if len(res)>0:
        bucket.delete_objects(Delete={'Objects': res})
```


```python
check_buckets(s3_res,return_names=True)
```

    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')





    ['data-enrichment-belt-exam']




```python

```


```python
## Create a bucket using resource 
import time
BUCKET_NAME = 'data-enrichment-belt-exam'


## CHecking if bucket already exists, if so, empty and delete
if BUCKET_NAME in check_buckets(s3_res,return_names=True):
     
    print(f"\n[!] BUCKET {BUCKET_NAME} ALREADY EXISTS. DELETING...")
    delete_all_objects(s3_res, BUCKET_NAME)
    time.sleep(5)

    s3_res.Bucket(BUCKET_NAME).delete()
    
    print('BUCKET DELETED.')
    
## Create bucket
bucket = s3_res.create_bucket(Bucket=BUCKET_NAME,
                              CreateBucketConfiguration={
                                  'LocationConstraint': session.region_name})
check_buckets(s3_res)
```

    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')
    
    [!] BUCKET data-enrichment-belt-exam ALREADY EXISTS. DELETING...
    BUCKET DELETED.
    [i] All Current Buckets:
    - s3.Bucket(name='data-enrichment-belt-exam')



```python
bucket
```




    s3.Bucket(name='data-enrichment-belt-exam')



### Getting Local Files to Upload


```python
## get list of files to upload
import glob,os
```


```python
upload_list = glob.glob("./exported_data/Northwind/*.csv")
upload_list
```




    ['./exported_data/Northwind/categories.csv',
     './exported_data/Northwind/products.csv',
     './exported_data/Northwind/OrderDetail.csv',
     './exported_data/Northwind/Order.csv']




```python
# demoing basename
os.path.basename(upload_list[0])
```




    'categories.csv'




```python
## Uploading files programmatically
for file in upload_list:
    key = os.path.basename(file)
    bucket.upload_file(file,key)
    print(f"- {file} saved as {key}")
```

    - ./exported_data/Northwind/categories.csv saved as categories.csv
    - ./exported_data/Northwind/products.csv saved as products.csv
    - ./exported_data/Northwind/OrderDetail.csv saved as OrderDetail.csv
    - ./exported_data/Northwind/Order.csv saved as Order.csv



```python
## Printing all files in bucket
[print(a) for a in bucket.objects.all()]
```

    s3.ObjectSummary(bucket_name='data-enrichment-belt-exam', key='Order.csv')
    s3.ObjectSummary(bucket_name='data-enrichment-belt-exam', key='OrderDetail.csv')
    s3.ObjectSummary(bucket_name='data-enrichment-belt-exam', key='categories.csv')
    s3.ObjectSummary(bucket_name='data-enrichment-belt-exam', key='products.csv')





    [None, None, None, None]




```python
## saving list of file objects
file_objects = [b for b in bucket.objects.all()]
file_obj = file_objects[0]#
file_obj.key
```




    'Order.csv'




```python
check_methods_attributes(file_obj)
```

    Attributes and Methods:
    - self.Acl
    - self.Bucket
    - self.MultipartUpload
    - self.Object
    - self.Version
    - self.bucket_name
    - self.copy_from
    - self.delete
    - self.e_tag
    - self.get
    - self.get_available_subresources
    - self.initiate_multipart_upload
    - self.key
    - self.last_modified
    - self.load
    - self.meta
    - self.owner
    - self.put
    - self.restore_object
    - self.size
    - self.storage_class
    - self.wait_until_exists
    - self.wait_until_not_exists

