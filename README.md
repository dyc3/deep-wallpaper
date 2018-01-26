# deep-wallpaper
Generate high resolution images using an ACGAN. Intended to make cool wallpapers.

I created this project for my Advanced Studies class in my Senior year of high school. 
The original idea was to take art from the internet, train a GAN to generate unique and
interesting art.

# Anatomy

This project uses 2 different models. One to generate images, and another to increase the 
resolution. The generator is an ACGAN, and the upscaler is a DCNN.

# Training

First, you need images.

`./data-collector.py`

Next, you need to sort the images into "good" or "bad" folders.

`./cleaner-tool.py`

After that, you need to give your images labels.

`./tagger-tool.py`

Finally, you can train the models.

`./main.py --train acgan`

`./main.py --train supersampler`

(NOTE: `--epochs COUNT` and `--resume EPOCH` are available)

Now you can generate images.

# Usage

I tried to make this project easy to run, but create an issue if you need help.

First, you need to install dependencies:

`pip install -r requirements.txt`

Then, you just need to run `main.py`. You can basically ignore all the other files
(unless you are training your own model), I've just left extra stuff in the 
project for archival purposes.

To print the help text:

`./main.py --help`

To resume training from a previous epoch:

`./main.py --train acgan --resume EPOCH`

To load the latest checkpoint and generate 1 image:

`./main.py --generate 1`

# Mistakes

When I started this project, I had nearly no knowledge of machine learning. After receiving
feedback from others, what I did wrong is pretty obvious to me now.

* The latent space of the generator has 100 dimensions, which is way too low. I plan on
retraining the model with about 1000-2000 dimensions in latent space, which will take a while.
* Some of my data is mislabeled. I will not be releasing the dataset until the issues with it
are fixed, which will also take a while.

## Further Improvements

* If we include the latent space as an input to the supersampler, it would probably allow
the supersampler to add more detail as it is upscaling the image.
