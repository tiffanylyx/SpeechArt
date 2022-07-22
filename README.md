
## Speech_Art Version
Update 2022/07/22

### Installation Instruction

Open the terminal to set up the virtual environment
```
conda create -n Speech_Art python=3.9
conda activate Speech_Art
```
Download the folder from github in your working path and cd to the folder
```
git clone https://github.com/tiffanylyx/Speech_Art.git
cd Speech_Art
```
Download all the models from the link https://drive.google.com/drive/folders/1EN_I-4xatD9rBqwbXmBfqHMoQjNW5nbv?usp=sharing.
Move all the models in the folder "Speech_Art/model".

Install all the dependencies
```
pip install -r requirements.txt
```
Install NLTK library
```
python install_nltk_file.py
```
Run the program
```
python generate_quad2.py
```
There will be a text input UI on the right-bottom corner of the window. Type in a sentence without special notions or numbers. Hit enter to see the result.


You might need to adjust the camera in the program to explore the generated 3D structure.

- Use the Right Mouse Button to zoom in and out.
- Use the Middle Mouse Button to rotate.
- Use the Left Mouse Button to pan.

### Update the codes
To keep the program updated, run
```
git pull
```

### Algorithm Rules
The 3D structures are generated based on the analytical results of the input sentence.
- One 3D structure = one sentence.
- One quad in one 3D structure = one word.
- Each structure represents one sentence. Each quad in the structure represents one word.
- The position of each 3D structure is decided by the sentence vector.
- The distance between the previous and the current structures represent the time period between the two input time
- Quads within one structure spread along the sentence vector.
- Two edges of the quad are decided by the word vectors. Then another two edges are connected to form a quad.
- The color of each quad (word) is decided by the word vector.
