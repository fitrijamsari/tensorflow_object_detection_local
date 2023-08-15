#count_label_every_subfolder.py is used to do some corrections on wrong label and count the total of correct labels
import xml.etree.ElementTree as ET
import os
import numpy as np
import difflib

gen_path = "/home/ofotechjkr/Desktop/dataset_count/dataset (copy)" #assign to respective folder containing the whole dataset

list1 = os.listdir(gen_path) #link to every file in the gen_path
print(list1) 

list_label = ["feeder_pillar", "bus_stop"] #correct name of the label
print(list_label)

all_list =[] # to store all labels found in .xml file
xml_list = [] # store only files with XML extension
similarity_list=[] #store the similarity rotio if the label is more than one
count_folder=0 #store the total folder
count_file=0 #store the total file

#function to extract the content of XML file
def get_root(path123, file123):
  tree = ET.parse(os.path.join(path123, file123))
  root =  tree.getroot()
  return tree, root

#function to correct the wrong label
def correction(path, file):
  tree, root12 = get_root(path, file)

#looping for every name inside a single XML file
  for name_ in root12.iter("name"):
    
    name__ = name_.text
    if name__ not in list_label: #check the availability of name inside the list 
      if len(list_label) ==1: #only one label
        new_name = list_label[0]
        name_.text = new_name #correct the wrong label
        tree.write(os.path.join(path, file))
      else:
        for ele in list_label: #more than one label
          similarity = difflib.SequenceMatcher(None, name__, ele).ratio() #producing string similarity ratio
          similarity_list.append(similarity)

        max_value = max(similarity_list) #find the index with the max ratio
        max_index = similarity_list.index(max_value)

        new_name = list_label[max_index] 
        name_.text = new_name #correct the wrong label
        tree.write(os.path.join(path, file))

#function to extract XML files
def xmlExtraction(path1234, file1234):
  print("extracting.....")
  _,root = get_root(path1234,file1234)
  for nameee in root.findall('object'):
    name = nameee.find('name').text
    all_list.append(name)
    

for every in list1:
  if os.path.isdir(os.path.join(gen_path, every)):
    print("I am directory")
    count_folder+=1

    list_sub = os.listdir(os.path.join(gen_path, every)) #link to every folder in the gen_path
    
    for every_sub in list_sub:
      print(str(list_sub) + "\n")

    #remove png extension and append the XML type
    for every_file in list_sub:
      _, ext = os.path.splitext(every_file)
      ext = ext[1:]
      if ext == "xml":
        xml_list.append(every_file)
    #print("Only XML Extensions---------------------->\n")
    for every_xml in xml_list:
      print(every_xml + "\n") 
    
    for file_ in xml_list:
      new_path = gen_path + "/" + every
      correction(new_path, file_)
      xmlExtraction(new_path,file_)
    
    xml_list.clear() # remove all ietms in the list for the next use

  else:
    print("I am file")
    count_file+=1
    _, ext = os.path.splitext(every)
    ext = ext[1:]
    if ext == "xml":
      correction(gen_path, every)
      xmlExtraction(gen_path,every)



#getting the number of unique label from the whole list
def unique(list1):
    x = np.array(list1)
    return np.unique(x)


print("\n")
print("####################COUNT LABEL SUMMARY################################################")
print("There are {} unique labels".format(len(unique(all_list))))

#counting the label with the respect of the number of unique label
c = [all_list.count(x) for x in all_list]

#transfer them into dictionary
d = dict(zip(all_list, c))

print("From this folder: "+ gen_path)
print("It has {} folders and {} files".format(count_folder,count_file))

print("The total labels from this folder: " + gen_path)
print(d)
print("#######################################################################################")