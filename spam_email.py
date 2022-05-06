import json
import boto3
import email
import ast
from email.parser import BytesParser
from email import policy
import email.utils
from email.utils import parseaddr
from hashlib import md5
import io
import sys
import numpy as np
import os

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def lambda_handler(event, context):
    
    # TODO implement

    print(event['Records'])
    s3 = boto3.client('s3')
    s1 = boto3.resource('s3')
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        file_name = str(record['s3']['object']['key'])

        # get the plain text version of the email
        msg = email.message_from_bytes(s3.get_object(Bucket=bucket_name, Key=file_name)['Body'].read())
        emailId = msg['From']
        date = msg['date']
        subject = msg['subject']
        # print(msg)
        print(emailId)
        print(date)
        print(subject)
        # print(email.utils.getaddresses([msg2['From']])[1])
        
        if msg.is_multipart():
            print("multi part")
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
    
            # skip any text/plain (txt) attachments
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    body = part.get_payload(decode=True)  # decode
                    print("multi part", body)
                    break
        else:
            #print("msg body is = ", msg.get_payload())
            body = msg.get_payload()
            
        body = body.decode("utf-8")
        email_msg = body.replace('\r\n','<br>').strip()
        print("INCOMING EMAIL BODY: " + email_msg)
        body = body.replace('\r\n',' ').strip()
        
        # call sagemaker api
        plain = [body]
        vocabulary_length = 9013
        one_hot_test_messages = one_hot_encode(plain, vocabulary_length)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
        payload = json.dumps(encoded_test_messages.tolist())
        prediction = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType = 'application/json', Body=payload)
        print("prediction is "+ str(prediction))
        print("prediction type is "+ str(type(prediction)))

        result = json.loads(prediction["Body"].read().decode())
        print(result)
        pred = int(result.get('predicted_label')[0][0])
        predicted_label = 'SPAM Email' if pred == 1 else 'HAM Email'
        score = str(float(result.get('predicted_probability')[0][0]) * 100)
        print("predicted answer is "+predicted_label)

        # call SES api to trigger email
         
        #email addresses    
        recipient_address = emailId
        sender_address = "noreply@freshmails.ml"
                    
        # The character encoding for the email.
        charset = "UTF-8"
                    
        #Region
        AWS_REGION = "us-east-1"
                
        # The subject line for the email.
        subject_email = "Spam or Ham Detection using AWS SES and Sagemaker"
              
        #The email body for recipients with non-HTML email clients.
        email_body= "We received your email sent at "+ date+ " with the subject <b>"+ subject + "</b>.<br>Here is a 240 character sample of the email body:<br>" + email_msg + "<br>The email was categorized as " + predicted_label + " with a " + score + "% confidence."
        #Create a new SES resource.
        client = boto3.client('ses',region_name=AWS_REGION)

        #try send an email

        try:
            response = client.send_email(
                Destination={
                    'ToAddresses': [
              recipient_address,
                                  ],
                            },
                            Message={
                             'Body': {
                                 'Html': {
                                     'Charset': charset,
                                     'Data': email_body,
                                 },
                             },
                             'Subject': {
                                 'Charset': charset,
                                 'Data': subject_email,
                             },
                         },
                         Source=sender_address,
                            
                     )
                    
        # Display an error if something goes wrong.	
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            print("Email sent! Message ID:",recipient_address),
            print(response['MessageId'])

def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    print(results)
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]