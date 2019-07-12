## Setup

### Dependencies

To run this you're going to need opencv-python

```sh
pip install opencv-python
```

### Images

Place a good copy in the root of this project and name it `correct.jpg`.

Make a folder called `images` and dump the scanned images into it.

## Running

Run `py alignimages.py` and it will scan all the images in `images/` and compare
them with `correct.jpg`. It will make a folder named `aligned` in `images/` and
place the aligned versions of the images there.
