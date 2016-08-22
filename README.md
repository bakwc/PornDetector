# PornDetector
Two python porn images (nudity) detectors.

First one (pcr.py) use scikit-learn and opencv. I was able to get ~85% accuracy on markup with 1500 positive and 1500 negative samples. It use two machine-learned classifiers - one of them use HSV colors histogram, and another use SIFT descriptors.

Second one (nnpcr.py) uses tensorflow neural network. I was able to get ~90% accuracy on the same markup. It use 4 convolutional (3x3 filters) combined with max_pool (2x2) layers, one 1024 fully connected layer and a softmax classifier at the end.

### Requirements of opencv & sklearn detector
- python 2.7
- scikit-learn 0.15
- opencv 2.4 (build it from sources, cause it [missing SIFT](http://stackoverflow.com/questions/18561910/opencv-python-cant-use-surf-sift) by default)

### Requirements of tensorlflow detector
- python 2.7
- opencv 2.4 (you can take binary from repository)
- latest tensorflow

This is my configuration, may be it can work with another library versions.

### Usage of opencv & sklearn detector
- Url prediction demo: `./pcr.py url http://example.com/img.jpg`
- Code usage:
```python
from pcr import PCR
model = PCR()
model.loadModel('model.bin')
predictions = model.predict(['image1.jpg', 'image2.jpg', 'image3.jpg'])
print predictions
```

### Usage of tensorlflow detector
- Url prediction demo: `./nnpcr.py url http://example.com/img.jpg`
- Code usage:
```python
from nnpcr import NNPCR
model = NNPCR()
model.loadModel('nnmodel.bin')
predictions = model.predict(['image1.jpg', 'image2.jpg', 'image3.jpg'])
print predictions
```

### Train model
- create directory 1 (with non-porn images), 2 (with porn images), cache (empty)
- Run `./pcr.py train` (to train opencv & sklearn) or `./nnpcr.py train` (for tensorflow one).

After train finish you will see accuracy and you will get "model.bin" file with your trained model. Now you can use it to detect porn (see functions predictTest and predictUrl). I added a sample model (model.bin) - you can test it without training your own model, but I recomend you to gather some huge collection of images (eg, 50K) for best results.

### License
Public domain (but it may use some patented algorithms, eg. SIFT - so you should check license of all used libraries).
