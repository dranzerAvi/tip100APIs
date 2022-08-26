# import main Flask class and request object
from flask import Flask, request, jsonify
from web3 import Web3
from flask_cors import CORS, cross_origin

import ssl
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import urllib.request as urllib2  # the lib that handles the url stuff
import random
import json
# import urllib.request
# import re
import warnings
# Importing for blockchain database apis
import datetime
import hashlib
import requests
## ML Model Api Imports

# Import the required libraries.
import os
import cv2
#import pafy
import math
import random
import numpy as np
import datetime as dt
# import tensorflow as tf
from collections import deque
# from tensorflow import keras

# from moviepy.editor import *
# %matplotlib inline

import joblib
import pandas as pd
import tensorflow
import numpy as np 
# import matplotlib
import math
import os
# from matplotlib import pyplot as plt
import IPython.display as ipd

import librosa
import librosa.display
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import IPython.display as ipd
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.optimizers import adam_v2
from tensorflow.keras.utils import to_categorical
import numpy as np 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize

ssl._create_default_https_context = ssl._create_unverified_context


#translator=Translator()
#from translate import Translator
warnings.filterwarnings("ignore")

# create the Flask app
app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
cred = credentials.Certificate('./key.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()


LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()`~-=_+[]{}|;\':",./<>? '

def create_cypher_dictionary():
    numbers = [ '%02d' % i for i in range(100) ]
    random.shuffle( numbers )
    return { a : b for a,b in zip( LETTERS, numbers ) }

def encrypt( cypher, string ) :
    return ''.join( cypher[ch] for ch in string )

def decrypt( cypher, string ) :
    inverse_cypher = { b : a for a,b in cypher.items() }
    return ''.join( inverse_cypher[a+b] for a,b in zip(*[iter(string)]*2) )

cypher = {'a': '75', 'b': '35', 'c': '21', 'd': '07', 'e': '99', 'f': '00', 'g': '03', 'h': '79', 'i': '90', 'j': '50', 'k': '85', 'l': '22', 'm': '26', 'n': '36', 'o': '17', 'p': '08', 'q': '82', 'r': '69', 's': '73', 't': '16', 'u': '28', 'v': '29', 'w': '45', 'x': '05', 'y': '27', 'z': '95', 'A': '89', 'B': '66', 'C': '42', 'D': '54', 'E': '46', 'F': '67', 'G': '12', 'H': '25', 'I': '93', 'J': '96', 'K': '56', 'L': '94', 'M': '11', 'N': '53', 'O': '39', 'P': '34', 'Q': '77', 'R': '32', 'S': '13', 'T': '37', 'U': '61', 'V': '58', 'W': '15', 'X': '63', 'Y': '97', 'Z': '68', '1': '47', '2': '64', '3': '24', '4': '44', '5': '78', '6': '84', '7': '19', '8': '62', '9': '09', '0': '10', '!': '88', '@': '23', '#': '41', '$': '74', '%': '51', '^': '60', '&': '06', '*': '14', '(': '20', ')': '72', '`': '04', '~': '91', '-': '57', '=': '52', '_': '59', '+': '98', '[': '71', ']': '81', '{': '76', '}': '38', '|': '30', ';': '31', "'": '83', ':': '49', '"': '02', ',': '65', '.': '87', '/': '92', '<': '55', '>': '40', '?': '80', ' ': '18'}
@app.route('/')
@cross_origin()
def form_example():
    return 'Form Data Example'

@app.route('/register', methods=['POST'])
@cross_origin()
def register():


# use the key and encrypt pwd
    requestData = json.loads(request.data)
    # passkey = 'TIP100'
 
    # str_encoded  = encrypt( cypher, phrase)

    
    
    users_ref = db.collection('users')
    infura_url="https://mainnet.infura.io/v3/5e840ebea3974bb18f3783d9c5a5c559"
    web3=Web3(Web3.HTTPProvider(infura_url))
    account=web3.eth.account.create()
    
    keystore=account.encrypt(requestData['phrase'])
    
    users_ref.document(keystore.get('address')).set({'keystore':keystore})

    return 'Registration Successful'

@app.route('/getUserID', methods=['POST'])
@cross_origin()
def getUserID():
    # return b'\xde\xad\xbe\xef'.hex()
    
    try: 
        
       
            # decoded=decrypt( cypher, doc.id )
            # if(decoded==phrase):
            #     userDoc=doc

        infura_url="https://mainnet.infura.io/v3/5e840ebea3974bb18f3783d9c5a5c559"
        web3=Web3(Web3.HTTPProvider(infura_url))
        requestData = json.loads(request.data)
        phrase = requestData['phrase']
            
        
        docs = db.collection(u'users').stream()
        

            
        for doc in docs:
            try:
                acc=web3.eth.account.decrypt(doc.get('keystore'),phrase)
                print(acc.hex())
                if acc:
                    account=doc.get('keystore')
            except:
                continue


    
        # account=web3.eth.account.decrypt(userDoc.get('keystore'),phrase)
        return account.get("address")
        
    except:
        return 'Account Not Found'

@app.route('/getAllTippers', methods=['GET'])
@cross_origin()
def getAllTippers():
    # return b'\xde\xad\xbe\xef'.hex()
    
    try: 
        
       
            # decoded=decrypt( cypher, doc.id )
            # if(decoded==phrase):
            #     userDoc=doc

        # infura_url="https://mainnet.infura.io/v3/5e840ebea3974bb18f3783d9c5a5c559"
        # web3=Web3(Web3.HTTPProvider(infura_url))
        # requestData = json.loads(request.data)
        # phrase = requestData['phrase']
            
        
        docs = db.collection(u'users').stream()
        users=[]

            
        for doc in docs:
            try:
                users.append({"uid":doc.get('keystore')['address'],"score":doc.get('score')})
                # print(acc.hex())
                # if acc:
                    # account=acc
            except:
                print('Exception')
                continue

    #     response = {'message': 'A block is MINED',
    #             'index': block['index'],
    #             'timestamp': block['timestamp'],
    #             'proof': block['proof'],
    #             'previous_hash': block['previous_hash']}
    # #  blockchain.to_json(block)
    #     return jsonify(response), 200
    
        # account=web3.eth.account.decrypt(userDoc.get('keystore'),phrase)
        return users
        
    except:
        return 'Account Not Found'


 
class Blockchain:
   
    # This function is created
    # to create the very first
    # block and set its hash to "0"
    def _init_(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0',crimeType='',description='',mediaURL='',urgency='',crimeTime='',score='',uid='',address='',dateOfIncident='',userScore='')
        docs = db.collection(u'tips').stream()   
        
        for doc in docs:
            try:
                self.create_block(proof=doc.get('proof'), previous_hash=doc.get('previous_hash'),crimeType=doc.get('crimeType'),description=doc.get('description'),mediaURL=doc.get('mediaURL'),urgency=doc.get('urgency'),crimeTime=doc.get('crimeTime'),score=doc.get('score'),uid=doc.get('uid'),address=doc.get('address'),dateOfIncident=doc.get('dateOfIncident'),userScore=doc.get('userScore'))
                print('Tip Added')
            except:
                print('Block creation Error')
                continue
        # if len(self.chain)==0:
        #     self.create_block(proof=1, previous_hash='0',crimeType='',description='',mediaURL='',urgency='',crimeTime='',score='',uid='',address='')
 
    # This function is created
    # to add further blocks
    # into the chain
    def create_block(self, proof, previous_hash,crimeType,description,mediaURL,urgency,crimeTime,score,uid,address,dateOfIncident,userScore):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'crimeType': crimeType,
                 'description': description,
                 'mediaURL': mediaURL,
                 'urgency': urgency,
                 'crimeTime': crimeTime,
                 'score': score,
                 'uid': uid,
                 'address': address,
                 'dateOfIncident': dateOfIncident,
                 'userScore': userScore,
                 'previous_hash': previous_hash,'isAlert':0,'view':1,'useful':0,'fake':0,'suspectScore':0,'mentalScore':0,'mediaScore':0,'likelyhood':0}
        self.chain.append(block)

        return block
       
    # This function is created
    
    # to display the previous block
    def print_previous_block(self):
        return self.chain[-1]
       
    # This is the function for proof of work
    # and used to successfully mine the block
    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
         
        while check_proof is False:
            hash_operation = hashlib.sha256(
                str(new_proof*2 - previous_proof*2).encode()).hexdigest()
            if hash_operation[:5] == '00000':
                check_proof = True
            else:
                new_proof += 1
                 
        return new_proof
 
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def to_json(self, block):
        blockJson = json.dumps(block)
        return blockJson
    
    # def upload_block(self,block):
    #     tips_ref = db.collection('tips')
    #     tips_ref.document(f"{len(self.chain)}").set(block)
  
    def chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
         
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
               
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(
                str(proof*2 - previous_proof*2).encode()).hexdigest()
             
            if hash_operation[:5] != '00000':
                return False
            previous_block = block
            block_index += 1
         
        return True
 

# Create the object
# of the class blockchain
blockchain = Blockchain()
 
# Mining a new block
@app.route('/addTip', methods=['POST'])
@cross_origin()
def addTip():
    requestData = json.loads(request.data)
    crimeType = requestData['crimeType']
    description = requestData['description']
    mediaURL = requestData['mediaURL']
    urgency = requestData['urgency']
    crimeTime = requestData['crimeTime']
    score = requestData['score']
    uid = requestData['uid']
    address = requestData['address']
    dateOfIncident = requestData['dateOfIncident']
    userScore = requestData['userScore']
    previous_block = blockchain.print_previous_block()
    previous_proof = previous_block['proof']
    # isA = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    block = blockchain.create_block(proof, previous_hash,crimeType,description,mediaURL,urgency,crimeTime,score,uid,address,dateOfIncident,userScore)
    # blockchain.upload_block(blockchain.to_json(block))
    tips_ref = db.collection('tips')
    blockJson=blockchain.to_json(block)
    tips_ref.document(f"{len(blockchain.chain)}").set(json.loads(blockJson))
    print(blockchain.to_json(block))
    response = {'message': 'A block is MINED',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash']}
    #  blockchain.to_json(block)
    return jsonify(response), 200
 
# Display blockchain in json format
@app.route('/getAllTips', methods=['GET'])
@cross_origin()
def getAllTips():
    allTips=[]
    docs = db.collection(u'tips').stream()   
        
    for doc in docs:
        # allTips.append(doc.to_dict())
        try:
            allTips.append(doc.to_dict())
            print('Tip Added')
        except:
            print('Block adding Error')
            continue
    response = {'chain': allTips,
                'length': len(allTips)}
    return jsonify(response), 200

@app.route('/getUserTips', methods=['GET'])
@cross_origin()
def getUserTips():
    userTips=[]
    # requestData = json.loads(request.data)
    # userID=requestData['uid']
    uid = request.args.get("uid")
    print(blockchain.chain)
    docs = db.collection(u'tips').stream()   
        
    for doc in docs:
        try:
            if(doc.get("uid")==uid):
                userTips.append(doc.to_dict())
                print('Tip Added')
        except:
            print('Block adding Error')
            continue
    # response = {'chain': allTips,
    #             'length': len(allTips)}
    # for block in blockchain.chain:
    #     if block['uid']==uid:
    #         userTips.append(block)
    response = {'chain': userTips,
                'length': len(userTips)}
    return jsonify(response), 200
 
# Check validity of blockchain
@app.route('/valid', methods=['GET'])
@cross_origin()
def valid():
    valid = blockchain.chain_valid(blockchain.chain)
     
    if valid:
        response = {'message': 'The Blockchain is valid.'}
    else:
        response = {'message': 'The Blockchain is not valid.'}
    return jsonify(response), 200

@app.route('/getTipDetails', methods=['GET'])
@cross_origin()
def getTipDetails():


    n = request.args.get("index")

    
    # requestData = json.loads(request.data)
    # userID=requestData['uid']
    
    print(blockchain.chain)
    docs = db.collection(u'tips').stream()   
        
    for doc in docs:
        try:
            if(doc.get("index")==n):
                
                print('Tip Added')
                return doc.to_dict()
        except:
            print('Block adding Error')
            continue
    
    # requestData = json.loads(request.data)
    # print(blockchain.chain)
    # print(requestData['index'])
    # response = {'chain': userTips,
    #             'length': len(userTips)}
    return [block for block in blockchain.chain if block and str(block['index'])==n], 200
LRCN_model = keras.models.load_model('./LRCNModel.h5')
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["Abuse", "Arrest", "Assault", "Burglary", "Explosion", "Fighting", "Normal Videos"]
def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    # print()
        
    # Release the VideoCapture object. 
    return [predicted_class_name,predicted_labels_probabilities[predicted_label]]
    # return f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}'








## ML Model API
@app.route('/get_Video_Score', methods=['POST'])
@cross_origin()
def get_Video_Score():


    requestData = json.loads(request.data)
    # # Download the youtube video.
    # # video_title = download_youtube_videos('https://youtu.be/fc3w827kwyA', test_videos_directory)
   
    # # Construct tihe nput youtube video path
    input_video_file_path = requestData['url']

    # Perform Single Prediction on the Test Video.
    score=predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
    return score
    # Display the input video.
    # VideoFileClip(input_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()

def get_Video_Score_func(input_video_file_path):


    # requestData = json.loads(request.data)
    # # Download the youtube video.
    # # video_title = download_youtube_videos('https://youtu.be/fc3w827kwyA', test_videos_directory)
   
    # # Construct tihe nput youtube video path
    # input_video_file_path = requestData['url']

    # Perform Single Prediction on the Test Video.
    score=predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
    return score

loaded_model = keras.models.load_model('./statewise_model.h5')

dictionary={'andhra pradesh': 2, 'adilabad': 3, 'anantapur': 35, 'chittoor': 184, 'cuddapah': 204, 'east godavari': 261, 'guntakal rly.': 326, 'guntur': 327, 'hyderabad city': 359, 'karimnagar': 454, 'khammam': 469, 'krishna': 514, 'kurnool': 520, 'mahaboobnagar': 552, 'medak': 576, 'nalgonda': 623, 'nellore': 639, 'nizamabad': 643, 'prakasham': 693, 'ranga reddy': 737, 'secunderabad rly.': 781, 'srikakulam': 840, 'vijayawada': 927, 'vijayawada rly.': 930, 'visakha rural': 934, 'visakhapatnam': 935, 'vizianagaram': 936, 'warangal': 941, 'west godavari': 947, 'total': 887, 'arunachal pradesh': 3, 'changlang': 168, 'dibang valley': 249, 'kameng east': 435, 'kameng west': 436, 'lohit': 537, 'papum pare': 669, 'siang east': 798, 'siang upper': 799, 'siang west': 800, 'subansiri lower': 848, 'subansiri upper': 849, 'tawang': 862, 'tirap': 882, 'assam': 4, 'barpeta': 87, 'bongaigaon': 137, 'c.i.d.': 151, 'cachar': 152, 'darrang': 220, 'dhemaji': 244, 'dhubri': 247, 'dibrugarh': 250, 'g.r.p.': 284, 'goalpara': 311, 'golaghat': 313, 'guwahati city': 331, 'hailakandi': 333, 'jorhat': 424, 'kamrup': 437, 'karbi anglong': 451, 'karimganj': 453, 'kokrajhar': 493, 'lakhimpur': 528, 'morigaon': 590, 'n.c.hills': 610, 'nagaon': 613, 'nalbari': 622, 'r.p.o.': 707, 'sibsagar': 801, 'sonitpur': 827, 'tinsukia': 881, 'bihar': 5, 'araria': 42, 'arwal': 45, 'aurangabad': 50, 'bagaha': 58, 'banka': 77, 'begusarai': 96, 'bettiah': 106, 'bhabhua': 108, 'bhagalpur': 110, 'bhojpur': 120, 'buxar': 148, 'darbhanga': 218, 'gaya': 306, 'gopalganj': 317, 'jamalpur rly.': 398, 'jamui': 405, 'jehanabad': 409, 'katihar': 460, 'katihar rly.': 462, 'khagaria': 468, 'kishanganj': 488, 'lakhisarai': 529, 'madhepura': 545, 'madhubani': 546, 'motihari': 591, 'munger': 599, 'muzaffarpur': 602, 'muzaffarpur rly.': 604, 'nalanda': 621, 'naugachia': 632, 'nawadah': 635, 'patna': 679, 'patna rly.': 681, 'purnea': 705, 'rohtas': 746, 'saharsa': 755, 'samastipur': 762, 'saran': 770, 'sheikhpura': 790, 'sheohar': 791, 'sitamarhi': 817, 'siwan': 820, 'supaul': 853, 'vaishali': 920, 'chhattisgarh': 7, 'balrampur': 71, 'bilaspur': 130, 'bizapur': 133, 'dantewara': 217, 'dhamtari': 235, 'durg': 259, 'grp raipur': 320, 'jagdalpur': 377, 'janjgir': 406, 'jashpur': 407, 'kanker': 441, 'kawardha': 465, 'korba': 504, 'koriya': 505, 'mahasamund': 554, 'raigarh': 711, 'raipur': 720, 'rajnandgaon': 727, 'sarguja': 771, 'goa': 12, 'north goa': 648, 'south goa': 832, 'gujarat': 13, 'ahmedabad commr.': 7, 'ahmedabad rural': 8, 'ahwa-dang': 10, 'amreli': 30, 'anand': 34, 'bharuch': 113, 'bhavnagar': 115, 'dahod': 210, 'gandhinagar': 294, 'himatnagar': 346, 'jamnagar': 400, 'junagadh': 425, 'kheda north': 481, 'kutch': 523, 'mehsana': 578, 'narmada': 628, 'navsari': 634, 'palanpur': 660, 'panchmahal': 665, 'patan': 675, 'porbandar': 690, 'rajkot commr.': 725, 'rajkot rural': 726, 'surat commr.': 856, 'surat rural': 857, 'surendranagar': 858, 'vadodara commr.': 918, 'vadodara rural': 919, 'valsad': 921, 'w.rly': 937, 'haryana': 14, 'ambala': 21, 'bhiwani': 119, 'faridabad': 271, 'fatehabad': 273, 'grp': 319, 'gurgaon': 330, 'hissar': 348, 'jhajjar': 411, 'jind': 418, 'kaithal': 432, 'karnal': 455, 'kurukshetra': 521, 'mahendragarh': 556, 'panchkula': 664, 'panipat': 667, 'rewari': 743, 'rohtak': 745, 'sirsa': 816, 'sonipat': 826, 'yamunanagar': 950, 'himachal pradesh': 15, 'chamba': 160, 'hamirpur': 334, 'kangra': 440, 'kinnaur': 486, 'kullu': 518, 'lahaul-spiti': 527, 'mandi': 566, 'shimla': 793, 'sirmaur': 814, 'solan': 821, 'una': 909, 'jammu & kashmir': 16, 'anantnag': 36, 'awantipora': 53, 'baramulla': 82, 'border': 138, 'budgam': 142, 'crime jammu': 199, 'crime srinagar': 201, 'doda': 256, 'ganderbal': 293, 'jammu': 399, 'kargil': 452, 'kathua': 459, 'kulgam': 517, 'kupwara': 519, 'leh': 535, 'poonch': 689, 'pulwama': 697, 'railways': 715, 'rajouri': 728, 'ramban': 732, 'reasi': 741, 'srinagar': 841, 'udhampur': 902, 'jharkhand': 17, 'bokaro': 135, 'chaibasa': 158, 'chatra': 169, 'deoghar': 230, 'dhanbad': 236, 'dhanbad rly.': 238, 'dumka': 257, 'garhwa': 297, 'giridih': 310, 'godda': 312, 'gumla': 323, 'hazaribagh': 345, 'jamshedpur': 401, 'jamshedpur rly.': 403, 'jamtara': 404, 'koderma': 491, 'latehar': 532, 'lohardagga': 536, 'pakur': 657, 'palamu': 659, 'ranchi': 736, 'sahebganj': 756, 'saraikela': 769, 'simdega': 810, 'karnataka': 18, 'bagalkot': 59, 'bangalore commr.': 75, 'bangalore rural': 76, 'belgaum': 99, 'bellary': 100, 'bidar': 124, 'bijapur': 127, 'chamarajnagar': 159, 'chickmagalur': 177, 'chitradurga': 181, 'dakshin kannada': 212, 'davanagere': 224, 'dharwad commr.': 242, 'dharwad rural': 243, 'gadag': 290, 'gulbarga': 322, 'hassan': 342, 'haveri': 344, 'k.g.f.': 426, 'kodagu': 490, 'kolar': 494, 'koppal': 502, 'mandya': 569, 'mysore commr.': 605, 'mysore rural': 606, 'raichur': 709, 'shimoga': 794, 'tumkur': 899, 'udupi': 904, 'uttar kannada': 914, 'kerala': 19, 'alapuzha': 14, 'cbcid': 155, 'ernakulam': 264, 'idukki': 362, 'kannur': 443, 'kasargod': 457, 'kollam': 498, 'kottayam': 509, 'kozhikode': 511, 'malappuram': 561, 'palakkad': 658, 'pathanamthitta': 676, 'thrissur': 877, 'trivandrum': 894, 'wayanadu': 945, 'madhya pradesh': 21, 'balaghat': 64, 'barwani': 89, 'betul': 107, 'bhind': 118, 'bhopal': 121, 'bhopal rly.': 123, 'chhatarpur': 174, 'chhindwara': 175, 'damoh': 215, 'datiya': 222, 'dewas': 233, 'dhar': 239, 'dindori': 253, 'guna': 324, 'gwalior': 332, 'harda': 339, 'hoshangabad': 350, 'indore': 368, 'indore rly.': 370, 'jabalpur': 373, 'jabalpur rly.': 375, 'jhabua': 410, 'katni': 463, 'khandwa': 470, 'khargon': 474, 'mandla': 567, 'mandsaur': 568, 'morena': 589, 'narsinghpur': 629, 'neemuch': 638, 'panna': 668, 'raisen': 721, 'rajgarh': 723, 'ratlam': 738, 'rewa': 742, 'sagar': 753, 'satna': 775, 'seoni': 784, 'shahdol': 786, 'shajapur': 788, 'sheopur': 792, 'shivpuri': 795, 'sidhi': 803, 'sihore': 804, 'tikamgarh': 880, 'ujjain': 905, 'umariya': 908, 'vidisha': 924, 'maharashtra': 22, 'ahmednagar': 9, 'akola': 13, 'amravati commr.': 28, 'amravati rural': 29, 'aurangabad commr.': 51, 'aurangabad rural': 52, 'beed': 95, 'bhandara': 111, 'buldhana': 144, 'chandrapur': 167, 'dhule': 248, 'gadchiroli': 291, 'gondia': 316, 'hingoli': 347, 'jalgaon': 393, 'jalna': 394, 'kolhapur': 496, 'latur': 533, 'mumbai': 593, 'mumbai rly.': 596, 'nagpur commr.': 616, 'nagpur rly.': 618, 'nagpur rural': 619, 'nanded': 625, 'nandurbar': 626, 'nasik commr.': 630, 'nasik rural': 631, 'navi mumbai': 633, 'osmanabad': 654, 'parbhani': 672, 'pune commr.': 698, 'pune rly.': 700, 'pune rural': 701, 'raigad': 710, 'ratnagiri': 739, 'sangli': 766, 'satara': 774, 'sindhudurg': 811, 'solapur commr.': 822, 'solapur rural': 823, 'thane commr.': 864, 'thane rural': 865, 'wardha': 943, 'washim': 944, 'yavatmal': 951, 'manipur': 23, 'bishnupur': 132, 'chandel': 164, 'churachandpur': 186, 'imphal(east)': 366, 'imphal(west)': 367, 'senapati': 783, 'tamenglong': 859, 'thoubal': 876, 'ukhrul': 906, 'meghalaya': 24, 'garo hills east': 299, 'garo hills south': 301, 'garo hills west': 304, 'jaintia hills': 379, 'khasi hills east': 476, 'khasi hills west': 479, 'ri-bhoi': 744, 'mizoram': 25, 'aizawl': 11, 'champhai': 163, 'kolasib': 495, 'lawngtlai': 534, 'lunglei': 544, 'mamit': 565, 'saiha': 757, 'serchhip': 785, 'nagaland': 26, 'dimapur': 251, 'kiphire': 487, 'kohima': 492, 'mokokchung': 585, 'mon': 586, 'peren': 684, 'phek': 685, 'tuensang': 897, 'wokha': 948, 'zunheboto': 952, 'odisha': 27, 'angul': 38, 'balasore': 65, 'baragarh': 81, 'berhampur': 105, 'bhadrak': 109, 'bolangir': 136, 'boudh': 141, 'cuttack': 205, 'deogarh': 229, 'dhenkanal': 245, 'gajapati': 292, 'ganjam': 296, 'jagatsinghpur': 376, 'jajpur': 389, 'jharsuguda': 416, 'kalahandi': 434, 'kandhamal': 439, 'kendrapara': 466, 'keonjhar': 467, 'khurda': 485, 'koraput': 503, 'malkangir': 563, 'mayurbhanj': 575, 'nayagarh': 637, 'nowrangpur': 652, 'nuapada': 653, 'puri': 704, 'rayagada': 740, 'rourkela': 748, 'sambalpur': 764, 'sonepur': 825, 'srp(cuttack)': 844, 'srp(rourkela)': 845, 'sundargarh': 852, 'punjab': 29, 'amritsar': 31, 'barnala': 86, 'batala': 92, 'bhatinda': 114, 'faridkot': 272, 'fatehgarh sahib': 275, 'ferozepur': 278, 'g.r.p': 283, 'gurdaspur': 329, 'hoshiarpur': 351, 'jagraon': 378, 'jalandhar': 390, 'kapurthala': 448, 'khanna': 471, 'ludhiana': 542, 'majitha': 560, 'mansa': 572, 'moga': 584, 'muktsar': 592, 'nawan shahr': 636, 'patiala': 678, 'ropar': 747, 'sangrur': 767, 'tarn taran': 861, 'rajasthan': 30, 'ajmer': 12, 'alwar': 20, 'banswara': 79, 'baran': 83, 'barmer': 85, 'bharatpur': 112, 'bhilwara': 116, 'bikaner': 129, 'bundi': 145, 'chittorgarh': 185, 'churu': 187, 'dausa': 223, 'dholpur': 246, 'dungarpur': 258, 'ganganagar': 295, 'hanumangarh': 337, 'jaipur': 382, 'jaisalmer': 388, 'jalore': 395, 'jhalawar': 412, 'jhunjhunu': 417, 'jodhpur': 419, 'karauli': 450, 'kota': 506, 'nagaur': 615, 'pali': 662, 'rajsamand': 729, 'sawai madhopur': 776, 'sikar': 805, 'sirohi': 815, 'tonk': 886, 'udaipur': 900, 'sikkim': 31, 'east': 260, 'north': 644, 'south': 829, 'west': 946, 'tamil nadu': 32, 'ariyalur': 43, 'chengai': 170, 'chennai': 171, 'chennai rly.': 172, 'coimbatore rural': 192, 'coimbatore urban': 193, 'cuddalore': 203, 'dharmapuri': 240, 'dindigul': 252, 'erode': 267, 'kanchipuram': 438, 'kanyakumari': 447, 'karur': 456, 'madurai rural': 549, 'madurai urban': 550, 'nagapattinam': 614, 'namakkal': 624, 'nilgiris': 642, 'perambalur': 683, 'pudukottai': 696, 'ramnathapuram': 734, 'salem rural': 760, 'salem urban': 761, 'sivagangai': 819, 'thanjavur': 866, 'theni': 867, 'thirunelveli rural': 870, 'thirunelveli urban': 871, 'thiruvallur': 872, 'thiruvannamalai': 873, 'thiruvarur': 874, 'thoothugudi': 875, 'trichy rly.': 891, 'trichy rural': 892, 'trichy urban': 893, 'vellore': 923, 'villupuram': 931, 'virudhunagar': 933, 'tripura': 34, 'dhalai': 234, 'uttar pradesh': 35, 'agra': 5, 'aligarh': 15, 'allahabad': 18, 'ambedkar nagar': 26, 'auraiya': 49, 'azamgarh': 54, 'badaun': 55, 'baghpat': 61, 'bahraich': 62, 'ballia': 67, 'banda': 73, 'barabanki': 80, 'bareilly': 84, 'basti': 91, 'bijnor': 128, 'bulandshahar': 143, 'chandoli': 166, 'chitrakoot dham': 183, 'deoria': 231, 'etah': 268, 'etawah': 269, 'faizabad': 270, 'fatehgarh': 274, 'fatehpur': 276, 'firozabad': 280, 'gautambudh nagar': 305, 'ghaziabad': 307, 'ghazipur': 308, 'gonda': 315, 'gorakhpur': 318, 'hardoi': 340, 'hathras': 343, 'j.p.nagar': 372, 'jalaun': 392, 'jaunpur': 408, 'jhansi': 413, 'kannauj': 442, 'kanpur dehat': 444, 'kanpur nagar': 445, 'kaushambi': 464, 'khiri': 482, 'kushi nagar': 522, 'lalitpur': 531, 'lucknow': 541, 'maharajganj': 553, 'mahoba': 558, 'mainpuri': 559, 'mathura': 573, 'mau': 574, 'meerut': 577, 'mirzapur': 583, 'moradabad': 587, 'muzaffarnagar': 601, 'pilibhit': 686, 'pratapgarh': 694, 'raibareilly': 708, 'rampur': 735, 'saharanpur': 754, 'sant kabirnagar': 768, 'shahjahanpur': 787, 'shrawasti': 797, 'sidharthnagar': 802, 'sitapur': 818, 'sonbhadra': 824, 'st.ravidasnagar': 846, 'sultanpur': 851, 'unnao': 911, 'varanasi': 922, 'uttarakhand': 36, 'almora': 19, 'bageshwar': 60, 'chamoli': 161, 'champawat': 162, 'dehradun': 227, 'haridwar': 341, 'nainital': 620, 'pauri garhwal': 682, 'pithoragarh': 687, 'rudra prayag': 749, 'tehri garhwal': 863, 'udhamsingh nagar': 903, 'uttarkashi': 916, 'west bengal': 37, '24 parganas north': 0, '24 parganas south': 1, 'asansol': 46, 'bankura': 78, 'birbhum': 131, 'burdwan': 146, 'coochbehar': 194, 'dakshin dinajpur': 211, 'darjeeling': 219, 'hooghly': 349, 'howrah': 352, 'howrah city': 353, 'howrah g.r.p.': 355, 'jalpaiguri': 396, 'kolkata': 497, 'malda': 562, 'midnapur': 582, 'murshidabad': 600, 'nadia': 612, 'purulia': 706, 'sealdah g.r.p.': 779, 'siliguri g.r.p.': 807, 'uttar dinajpur': 913, 'a & n islands': 0, 'andaman': 37, 'nicobar': 641, 'chandigarh': 165, 'd & n haveli': 8, 'd and n haveli': 208, 'daman & diu': 10, 'daman': 214, 'diu': 255, 'delhi ut': 11, 'central': 157, 'delhi ut total': 228, 'g.r.p.(rly)': 287, 'i.g.i. airport': 361, 'new delhi': 640, 'north east': 647, 'north west': 649, 's.t.f.': 751, 'south west': 833, 'lakshadweep': 530, 'puducherry': 695, 'pondicherry': 688, 'kabirdham': 429, 'handwara': 336, 'kharagpur g.r.p.': 473, 'paschim midnapur': 674, 'purab midnapur': 703, 'spl cell': 836, 'cyberabad': 207, 'k/kumey': 428, 'upper dibang valley': 912, 'narayanpur': 627, 'ernakulam commr.': 265, 'ernakulam rural': 266, 'kozhikode commr.': 512, 'kozhikode rural': 513, 'trivandrum commr.': 895, 'trivandrum rural': 896, 'anuppur': 41, 'ashok nagar': 48, 'burhanpur': 147, 'krishnagiri': 515, 'baska': 90, 'chirang': 180, 'udalguri': 901, 'surajpur': 854, 'mewat': 581, 'ferozpur': 279, 'viluppuram': 932, 'igi airport': 363, 'north-east': 650, 'north-west': 651, 'south-west': 835, 'vijayawada city': 928, 'palwal': 663, 'border district': 139, 'mumbai commr.': 594, 'imphal east': 364, 'imphal west': 365, 'sas ngr': 773, 'jaipur east': 383, 'jaipur north': 384, 'jaipur rural': 385, 'jaipur south': 386, 'jodhpur city': 420, 'jodhpur rural': 422, 'kota city': 507, 'kota rural': 508, 'grp(rly)': 321, 'stf': 847, 'karaikal': 449, 'khunti': 484, 'ramgarh': 733, 'longleng': 539, 'amritsar rural': 32, 'a and n islands': 2, 'outer': 656, 'tapi': 860, 'bandipora': 74, 'kishtwar': 489, 'samba': 763, 'shopian': 796, 'cbpura': 156, 'ramanagar': 731, 'alirajpur': 17, 'singrauli': 812, 'cid': 188, 'dcp bbsr': 225, 'dcp ctc': 226, 'ludhiana rural': 543, 'sbs nagar': 777, 'chennaisuburban': 173, 'kanshiram nagar': 446, 'caw': 154, 'crime branch': 198, 'eow': 263, 'south-east': 834, 'anjaw': 39, 'baddipolicedist': 57, 'railways kmr': 719, 'tiruppur': 884, 'guntur urban': 328, 'rajahmundry': 722, 'tirupathi urban': 883, 'warangal urban': 942, 'rural': 750, 'cid crime': 189, 'crime kashmir': 200, 'sopore': 828, 'yadgiri': 949, 'cp amritsar': 195, 'cp jalandhar': 196, 'cp ludhiana': 197, 'jalandhar rural': 391, 'g.r.p. ajmer': 285, 'g.r.p. jodhpur': 286, 'csm nagar': 202, 'ramabai nagar': 730, 'gariyaband': 298, 'kutch (east(g))': 524, 'kutch (west-bhuj)': 525, 'w.rly ahmedabad': 939, 'w.rly vadodara': 940, 'railways jammu': 716, 'railways kashmir': 717, 'mangalore city': 570, 'kollam commr.': 499, 'kollam rural': 500, 'thrissur commr.': 878, 'thrissur rural': 879, 'fazilka': 277, 'pathankot': 677, 'g.r.p.ajmer': 288, 'g.r.p.jodhpur': 289, 'jaipur west': 387, 'bhim nagar': 117, 'panchshil nagar': 666, 'prabuddh nagar': 692, 'baksa': 63, 'bieo': 126, 'hamren': 335, 'balod': 68, 'baloda bazar': 69, 'bemetara': 101, 'kondagaon': 501, 'mungeli': 598, 'sukma': 850, 'ambala rural': 24, 'ambala urban': 25, 'railways katra': 718, 'jodhpur east': 421, 'jodhpur west': 423, 'gomati': 314, 'khowai': 483, 'sipahijala': 813, 'unakoti': 910, 'bdn cp': 94, 'bkp cp': 134, 'jhargram': 414, 'siliguri_pc': 809, 'car': 153, 'zz total': 953, 'longding': 538, 'i&p haryana': 360, 'agar': 4, 'garo hills north': 300, 'garo hills south w.': 302, 'jaintia hills east': 380, 'jaintia hills west': 381, 'khasi hills south w.': 477, 'spl narcotic': 837, 'traffic ps': 888, 'discom': 254, 'amethi': 27, 'amroha': 33, 'hapur': 338, 'kasganj': 458, 'sambhal': 765, 'shamli': 789, 'a&n islands': 1, 'd&n haveli': 209, 'metro rail': 580, 'guntakal railway': 325, 'vijayawada railway': 929, 'kukung kumey': 516, 'lower dibang valley': 540, 'papum pare city': 670, 'papum pare rural': 671, 'g. r. p.': 281, 'n. c. hills': 609, 'jamalpur railway': 397, 'katihar railway': 461, 'muzaffarpur railway': 603, 'patna railway': 680, 'economic offences unit': 262, 'anti terrorist squad': 40, 'balodbazar': 70, 'bemetra': 102, 'g. r. p. raipur': 282, 'mungali': 597, 'c. i. d.': 149, 'ahmedabad city': 6, 'arvalli': 44, 'banaskantha': 72, 'botad': 140, 'chhotaudepur': 176, 'c. i. d. crime': 150, 'dang': 216, 'devbhumi dwarka': 232, 'gir somnath': 309, 'kachchh east(g)': 430, 'kachchh west(b)': 431, 'kheda': 480, 'mahisagar': 557, 'morbi': 588, 'rajkot city': 724, 'sabarkantha': 752, 'surat city': 855, 'vadodara city': 917, 'w.rly  ahmedabad': 938, 'ambala (rural)': 22, 'ambala (urban)': 23, 'mahendergarh': 555, 'irrigation & power': 371, 'baddi': 56, 'lahaul & spiti': 526, 'dhanbad railway': 237, 'jamshedpur railway': 402, 'bengaluru city': 103, 'bengaluru district': 104, 'belagavi district': 98, 'ballari': 66, 'vijayapura': 926, 'chikkaballapura': 178, 'chikkamagaluru': 179, 'dakshina kannada': 213, 'dharwad': 241, 'kalaburgi': 433, 'hubballi dharwad city': 358, 'mangaluru city': 571, 'mysuru city': 607, 'mysuru district': 608, 'k.railways': 427, 'tumakuru': 898, 'uttara kannada': 915, 'belagavi city': 97, 'bhopal railway': 122, 'datia': 221, 'indore railway': 369, 'jabalpur railway': 374, 'khargone': 475, 'sehore': 782, 'umaria': 907, 'cyber cell': 206, 'mumbai railway': 595, 'nagpur railway': 617, 'palghar': 661, 'pune railway': 699, 'garo hills south west': 303, 'khasi hills south west': 478, 'spl traffic': 838, 'malkangiri': 564, 'nabarangpur': 611, 'srp (cuttack)': 842, 'srp (rourkela)': 843, 'bathinda': 93, 'sas nagar': 772, 'praapgarh': 691, 'coimbatore': 190, 'coimbatore city': 191, 'madurai': 547, 'madurai city': 548, 'railway chennai': 713, 'railway trichy': 714, 'salem': 758, 'salem city': 759, 'thirunelveli': 868, 'thirunelveli city': 869, 'tiruppur city': 885, 'trichy': 889, 'trichy city': 890, 'other units': 655, 'telangana': 33, 'mahaboob nagar': 551, 'secunderabad railway': 780, 'kowai': 510, 'chitrakoot': 182, 'north 24 parganas': 646, 'south 24 parganas': 830, 'alipurduar': 16, 'asansol-durgapur pc': 47, 'barrackpur pc': 88, 'bidhannagar pc': 125, 'howrah g. r. p.': 354, 'howrah pc': 356, 'howrah rural': 357, 'jhargram police district': 415, 'kharagpur g. r. p.': 472, 'paschim medinipur': 673, 'purab medinipur': 702, 'sealdah g. r. p.': 778, 'siliguri g. r. p.': 806, 'siliguri pc': 808, 'south andaman': 831, 'north & middle andaman': 645, 'metro': 579, 'railway': 712, 'spuwac': 839, 'vigilance': 925}
crime_list={'MURDER': 0,'ATTEMPT TO MURDER': 1,'RAPE': 2, 'KIDNAPPING': 3, 'ROBBERY':4,'VIOLENCE':5,'CHEATING':6,'CRIME ON WOMEN': 7, 'HUMAN TRAFFICKING': 8}
data=[['ANDHRA PRADESH','ADILABAD']]
test = pd.DataFrame(data, columns=['State_Label', 'District_Label'])
test['State_Label'] = test['State_Label'].str.lower()
test['District_Label'] = test['District_Label'].str.lower()



## ML Model API
@app.route('/get_State_Score', methods=['POST'])
@cross_origin()
def get_State_Score():
    requestData = json.loads(request.data)
    state= requestData['state']
    district= requestData['district']
    crime= requestData['crime']


    data=[[state,district]]
    test = pd.DataFrame(data, columns=['State_Label', 'District_Label'])
    test['State_Label'] = test['State_Label'].str.lower()
    test['District_Label'] = test['District_Label'].str.lower()
    # crime=[]
    if not test.iloc[0,0] in dictionary:
      return "0"
    else:
      test.iloc[0,0]=dictionary[test.iloc[0][0]]

    if not test.iloc[0,1] in dictionary:
      return "0"
    else:
      test.iloc[0,1]=dictionary[test.iloc[0][1]]
    test = test.apply(lambda col:pd.to_numeric(col, errors='coerce'))
    ans=loaded_model.predict(test,verbose=0)
    # print()  ##input
    return f"{ans[0][crime_list[crime]]}"



def get_State_Score_func(state,district,crime):
    # requestData = json.loads(request.data)
    # state= requestData['state']
    # district= requestData['district']
    # crime= requestData['crime']


    data=[[state,district]]
    test = pd.DataFrame(data, columns=['State_Label', 'District_Label'])
    test['State_Label'] = test['State_Label'].str.lower()
    test['District_Label'] = test['District_Label'].str.lower()
    # crime=[]
    if not test.iloc[0,0] in dictionary:
      return "0"
    else:
      test.iloc[0,0]=dictionary[test.iloc[0][0]]

    if not test.iloc[0,1] in dictionary:
      return "0"
    else:
      test.iloc[0,1]=dictionary[test.iloc[0][1]]
    test = test.apply(lambda col:pd.to_numeric(col, errors='coerce'))
    ans=loaded_model.predict(test,verbose=0)
    # print()  ##input
    return f"{ans[0][crime_list[crime]]}"



pipe_lr = joblib.load(open("./emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
sample=["I believe that the murderer of garvit murder case is jatin rawat"]


## ML Model API
@app.route('/get_Emotion', methods=['POST'])
@cross_origin()
def get_Emotion():
    requestData = json.loads(request.data)
    sample= requestData['desc']
    sample=[sample]
    lst=[]
    # print(pipe_lr.predict(sample)[0])
    # print(emotions_emoji_dict[pipe_lr.predict(sample)[0]])
    print(pipe_lr.predict_proba(sample))
    lst.append(pipe_lr.predict(sample)[0])
    lst.append(emotions_emoji_dict[pipe_lr.predict(sample)[0]])
    # lst.append(pipe_lr.predict_proba(sample)[0][0])
    response = {'list': lst}
    return jsonify(response), 200
    # return lst






def mp3tomfcc(file_path, max_pad):
  

  r = requests.get(file_path, allow_redirects=True)
  open('./___audio.wav', 'wb').write(r.content)
  audio, sample_rate = librosa.core.load("./___audio.wav")
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
  pad_width = max_pad - mfcc.shape[1]
  if (pad_width > 0):
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
  else:
    mfcc = mfcc[:,0:max_pad]
  return mfcc

model = tf.keras.models.load_model('./__truthorlie.h5')


## ML Model API
@app.route('/get_Audio_Score', methods=['POST'])
@cross_origin()
def get_Audio_Score(input_video_file_path):
   
    # input_video_file_path = "TIP100 Blockchain API\__trial_truth_012.wav"
    test=mp3tomfcc(input_video_file_path,1000)
    test=np.asarray(test)
    test=test[np.newaxis, :, :]
    ans=model.predict(test)
    if(ans[0][1]>ans[0][0]):
      print('LIE')
      return 'LIE'
    else:
      print('TRUTH')
      return 'TRUTH'


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

def ProperNounExtractor(text):
    lst=[]
    
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'NNP': # If the word is a proper noun
                lst.append(word)


    return lst

## ML Model API
@app.route('/get_PN', methods=['POST'])
@cross_origin()
def get_PN():
    requestData = json.loads(request.data)
    text= requestData['text']
    #New Commit
    # text =  "Rohan is a wonderful player. He was born in India. He is a fan of the movie Wolverine. He has a dog named Bruno."

# Calling the ProperNounExtractor function to extract all the proper nouns from the given text. 
    return ProperNounExtractor(text)   

def get_PN_func(text):
    # requestData = json.loads(request.data)
    # text= requestData['text']
    # text =  "Rohan is a wonderful player. He was born in India. He is a fan of the movie Wolverine. He has a dog named Bruno."
   
# Calling the ProperNounExtractor function to extract all the proper nouns from the given text. 
    return ProperNounExtractor(text)        
@app.route('/get_Trust_Score', methods=['POST'])
@cross_origin()
def get_Trust_Score():
    requestData = json.loads(request.data)
    urgency_score=0
    user_score=0
    mental_score=0

    text= requestData['text']
    state= requestData['state']
    district= requestData['district']
    crime= requestData['crime']
    input_audio_file_path = requestData['audio']
    input_video_file_path = requestData['video']
    text1=get_PN_func(text)
    pn_score=len(text1)/len(text)
    # pn_score=0
    audio_score=1
    if(len(input_audio_file_path)!=0):
        if(get_Audio_Score(input_audio_file_path)=='LIE'):
            audio_score=0
    
    
    state_score=float(get_State_Score_func(state,district,crime))
    # state_score=0
    if(len(input_video_file_path)==0):
        media_score=0
    else:
        media_score=get_Video_Score_func(input_video_file_path)[1]
    # media_score=0
    
    final_score= (28.2 * user_score) + (23.5*state_score) + (18.8*(media_score+audio_score)/2) + (14.1*mental_score) + (9.4* pn_score) +(4.7*urgency_score) 
    print(final_score)
    return f"{final_score}"



    







if __name__ == '__main__':
#     # run app in debug mode on port 5000
    app.run(debug=True, port=5000)