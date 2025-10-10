import requests, zipfile, io 
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import seaborn 


#link to the dataset
url = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"

#download the zip 
request = requests.get(url)
zipped = zipfile.ZipFile(io.BytesIO(request.content))
zipped.extractall(".")

file_path = r'KDD+.txt'
#define the column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]

#load the dataset inro a df
df = pandas.read_csv(file_path, names= columns)
#print(df.head(20))

#preprocess the data

df['attack_flag'] = df['attack'].apply(lambda x: 0 if x=='normal' else 1)
#print(df.head(20))

#not to miss out on granularity we have to add labels according to the type of attack
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 
                     'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 
                  'snmpguess', 'spy', 'warezclient', 'warezmaster', 
                  'xclock', 'xsnoop']

def map_attacks(attack):
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    else:
        return 0
    
df['map_attack']= df['attack'].apply(map_attacks)
#print(df.head(15))

#one-hot encoding of categorical features - ml algos only operate with numbers 
f= ['protocol_type', 'service']
encoded = pandas.get_dummies(df[f])
#print(encoded.head(20))

#numeric features (statistical measures of the traffic)
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate'
]
#features
train_set = encoded.join(df[numeric_features])
#desired output
multi_y = df['map_attack']

#split the data into training and test sets with a randomnes generator seed of 1337 
train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)

#divide the set further to tune the parameters (70/30)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)

# training the random forest classifier
rf = RandomForestClassifier( random_state=1337)
rf.fit(multi_train_X, multi_train_y)

multi_predictions = rf.predict(multi_val_X) #predict the y based on x
accuracy = accuracy_score(multi_val_y, multi_predictions) #compare the predicted y with the actual y 
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall = recall_score(multi_val_y, multi_predictions, average='weighted')
f1 = f1_score(multi_val_y, multi_predictions, average='weighted')
print(f"validation Set Evaluation:")
print(f"accuracy: {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1-score: {f1:.4f}")

# confusion matrix - how many elements of each class were classified as each possible class 
conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['normal', 'dos', 'probe', 'privilege', 'access']
seaborn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=class_labels, yticklabels=class_labels)

import matplotlib.pyplot as plt
plt.title('matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


print("classification report")
print("-"*40)
print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))

#testing on the 20% test set - completely unseen data separated at the beginning 
test_multi_predictions = rf.predict(test_X)
test_accuracy = accuracy_score(test_y, test_multi_predictions)
test_precision = precision_score(test_y, test_multi_predictions, average='weighted')
test_recall = recall_score(test_y, test_multi_predictions, average='weighted')
test_f1 = f1_score(test_y, test_multi_predictions, average='weighted')
print("\ntest set svaluation:")
print(f"accuracy: {test_accuracy:.4f}")
print(f"precision: {test_precision:.4f}")
print(f"recall: {test_recall:.4f}")
print(f"f1-score: {test_f1:.4f}")

#test set conf matrix
test_conf_matrix = confusion_matrix(test_y, test_multi_predictions)
seaborn.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=class_labels, yticklabels=class_labels)
plt.title('matrix of the training set')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

#report
print("report for the training set")
print(classification_report(test_y, test_multi_predictions, target_names=class_labels))

#saving in a .joblib
import joblib

# Save the trained model to a file
model_filename = 'model.joblib'
joblib.dump(rf, model_filename)

