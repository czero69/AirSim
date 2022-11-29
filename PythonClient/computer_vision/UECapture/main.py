from capture import Capture
import logging
import argparse

def main(args):
    logging.getLogger().setLevel(logging.INFO)

    be_verbose = args["verbose"]
    do_record = args["record"]
    camera_paths = args["camera_paths_dir"]
    save_dir = args["save_dir"]
    generate_camera_sample_dir = args["generate_camera_sequences_dir"]

    capture_interface = Capture(verbose=be_verbose, camera_paths=camera_paths)

    # set segmentation mask ids
    capture_interface.setSegmentationIDs()

    if do_record:
        # same time dilation must be set in Velocity Material to keep Velocity correct
        capture_interface.record(time_dilation=0.01)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_paths_dir",
        type=str,
        help="folder with camera paths/movements files",
        default="./camera_paths/slow_walking_1/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="save dir for generated sequences",
        default="./camera_paths/",
    )
    parser.add_argument(
        "--generate_camera_sequences_dir",
        type=str,
        help="generate cam sequences and save it to the dir",
        default=None
    )
    parser.add_argument(
        "--record",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="record cameras paths from capture dir",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="print more info",
    )
    args = vars(parser.parse_args())
    main(args)