# RestoBot
Files and Folders:
1. RESTOBOT.py – Main Python File
2. Create_Indexes.py – Python file to create indexes from data
3. /Data – Folder containing raw data (.pdf, .txt, .csv)
4. /Data_Indexes – Folder containing the created indexes
5. requirements.txt – A text file containing required libraries and their versions
How to run the Project:
1. Install all the necessary libraries (pip install …). All the required libraries and their versions are present
in requirements.txt
2. Open RESTOBOT.py in a text editor and replace <ENTER API> with the actual API tokens. Required APIs
are listed below.
(To skip this step and get a version with all the API tokens already present, please email our team at
nazirkar@usc.edu / dunbray@usc.edu / abhange@usc.edu)
3. Open a terminal in the directory having RESTOBOT.py and enter the following command:
streamlit run RESTOBOT.py
This will open our chatbot in a web browser and is ready to use :)
Required API tokens:
1. REPLICATE API (https://replicate.com/)
2. LOG MEAL API (https://logmeal.es/api)
3. CLIP DROP API (https://clipdrop.co/apis)
4. GOOGLE MAPS API (https://developers.google.com/maps)
Complete Workflow:
1. All the raw data files collected from various sources are stored in the /Data folder.
2. Run the Create_Indexes.py script using a terminal by entering the following command:
python3 Create_Indexes.py . This creates the /Data_Indexes directory, which contains multiple .json
files. These files are the vector indexes store for our data.
3. Finally, we follow the steps in ‘How to run the Project’.
