import requests

"""
Example Script for positing a file to the HoloServer
"""

url = 'http://holoserver:5000'                                  

survey = 'PostTest'                                                 # Name of the working survey
scan = '081'                                                        # Number of the current scan
file = 'FILE____081.DZT'                                            # DZT assocaited with the current scan

remote_path = f'{survey}/{scan}/{file}'                             # Path to save the DZT/CSV on the server
local_path = f'{file}'                                              # local path to the DZT/CSV

files = {'file': (remote_path, open(local_path, 'rb'))}             # Construct a dictionary for the post request

_ = requests.post(url, files=files)
