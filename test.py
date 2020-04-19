from absl import flags, app
from absl.flags import FLAGS


flags.DEFINE_boolean("cache_test",
                     True,
                     "Cache the test set in RAM or not")
