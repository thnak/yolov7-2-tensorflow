from tkinter.filedialog import askdirectory
  
# Importing required libraries.
from tkinter import Tk
import os
import hashlib
from pathlib import Path
import time
# We don't want the GUI window of
# tkinter to be appearing on our screen
Tk().withdraw()
  
# Dialog box for selecting a folder.
file_path = askdirectory(title="Select a folder")
  
# Listing out all the files
# inside our root folder.
list_of_files = os.walk(file_path)
  
# In order to detect the duplicate
# files we are going to define an empty dictionary.
unique_files = dict()
t0 = time.time()
for root, folders, files in list_of_files:
  
    # Running a for loop on all the files
    for file in files:
  
        # Finding complete file path
        file_path = Path(os.path.join(root, file))
  
        # Converting all the content of
        # our file into md5 hash.
        Hash_file = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
  
        # If file hash has already #
        # been added we'll simply delete that file
        if Hash_file not in unique_files:
            unique_files[Hash_file] = file_path
        else:
            os.remove(file_path)
            print(f"{file_path} has been deleted")
finishTime = time.time() - t0
finishTime = round(finishTime,2)
print("finish in",str(finishTime)+"s")