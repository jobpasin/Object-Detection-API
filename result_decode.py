import json
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('logs', './detections/report2.json', 'path to log file')

def main(_argv):
    with open(FLAGS.logs) as f:
        data = json.load(f)
    for k in data.keys():
        print(data[k])


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass