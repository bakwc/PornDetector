# PornDetector
Porn detector with python, scikit-learn and opencv. I was able to get ~90% accuracy on markup with 1500 positive and 1500 negative samples. It use two machine-learned classifiers - one of them use HSV colors histogram, and another use SIFT descriptors.

### Requirements
- python 2.7
- scikit-learn 0.15
- opencv 2.4 (build it from sources, cause it [missing SIFT](http://stackoverflow.com/questions/18561910/opencv-python-cant-use-surf-sift) by default)

This is my configuration, may be it can work with another library versions.

### Usage
- Url prediction demo: `./pcr.py url http://example.com/img.jpg`
- Code usage:
```python
from pcr import PCR
model = PCR()
model.loadModel('model.bin')
predictions = model.predict(['image1.jpg', 'image2.jpg', 'image3.jpg'])
print predictions
```

### Train model
- create directory 1 (with non-porn images), 2 (with porn images), cache (empty)
- Run `./pcr.py train`.

After train finish you will see accuracy and you will get "model.bin" file with your trained model. Now you can use it to detect porn (see functions predictTest and predictUrl). I added a sample model (model.bin) - you can test it without training your own model, but I recomend you to gather some huge collection of images (eg, 50K) for best results.

### License
Public domain (but it may use some patented algorithms, eg. SIFT - so you should check license of all used libraries).
