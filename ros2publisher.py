
import sys
import argparse
import rclpy
from rclpy.node import Node
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, logUsage


class ObjectDetectorNode(Node):
    def __init__(self, args):
        super().__init__('object_detector_node')
        self.args = args
        self.net = detectNet(self.args.network, sys.argv, self.args.threshold)
        self.input = videoSource(self.args.input_URI, argv=sys.argv)
        self.output = videoOutput(self.args.output_URI, argv=sys.argv)

        self.publisher_ = self.create_publisher(String, 'detections', 10)
        self.timer = self.create_timer(1.0 / self.args.fps, self.detect_objects)

    def detect_objects(self):
        img = self.input.Capture()
        detections = self.net.Detect(img, overlay=self.args.overlay)

        msg = String()
        msg.data = 'detected {:d} objects in image\n'.format(len(detections))
        for detection in detections:
            msg.data += str(detection) + '\n'

        self.publisher_.publish(msg)
        self.output.Render(img)
        self.output.SetStatus("{:s} | Network {:.0f} FPS".format(self.args.network, self.net.GetNetworkFPS()))
        self.net.PrintProfilerTimes()

        if not self.input.IsStreaming() or not self.output.IsStreaming():
            self.get_logger().info('Input or output stream ended')
            self.timer.cancel()
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Locate objects in a live camera stream using an object detection DNN.')
    parser.add_argument('input_URI', type=str, default='', nargs='?', help='URI of the input stream')
    parser.add_argument('output_URI', type=str, default='', nargs='?', help='URI of the output stream')
    parser.add_argument('--network', type=str, default='ssd-mobilenet-v2', help='pre-trained model to load (see below for options)')
    parser.add_argument('--overlay', type=str, default='box,labels,conf', help='detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  "box", "labels", "conf", "none"')
    parser.add_argument('--threshold', type=float, default=0.5, help='minimum detection threshold to use')
    parser.add_argument('--fps', type=int, default=30, help='maximum FPS to process')
    args = parser.parse_args(args=args)

    node = ObjectDetectorNode(args)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
