
import os


#FIRE_PATH = "first_try_fajer"
#NON_FIRE_PATH = "first_try_no_fajer"
FIRE_PATH = "fajer_flickr"
NON_FIRE_PATH = "no_fire_flickr"
FLICKR_PATH = "flickr30k_images/16"

CLASSES = ["Non-Fire", "Fire"]


TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25


INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 20

#MODEL_PATH = os.path.sep.join(["output", "first_try_fire_detection.model"])
#MODEL_PATH = os.path.sep.join(["output", "more_img_fire_detection.model"])
#MODEL_PATH = os.path.sep.join(["output", "more_epochs_fire_detection.model"])
MODEL_PATH = os.path.sep.join(["output", "best_fire_detection.model"])
#MODEL_PATH = os.path.sep.join(["output", "flickr_fire_detection.model"])



LRFIND_PLOT_PATH = os.path.sep.join(["output", "flickr_lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "flickr_training_plot.png"])


OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 3