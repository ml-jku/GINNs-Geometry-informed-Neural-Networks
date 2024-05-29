import os
import glob
import imageio
from PIL import Image
from natsort import natsorted


def animate(folder_path, duration=.05, loop=0):
    """
    Finds all pngs in the specified folder_path,
    orders them numerically and stores anim.gif to the same folder.
    """
    ## Use glob to search for PNG files in the specified folder
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    ## Sort the list of PNG files numerically, so eg prefix_10_suffix goes after prefix_5_suffix
    png_files = natsorted(png_files)
    for file in png_files:
        print(file)

    frames = [Image.open(png_file) for png_file in png_files]
    print(f'Read {len(frames)} frames')
    frame_durations = [duration for f in frames]
    imageio.mimsave(f'{folder_path}/anim.gif', frames, duration=frame_durations, loop=loop)
    # imageio.mimsave(f'{folder_path}/anim.gif', frames, duration=duration, loop=loop)


if __name__ == '__main__':
    # folder_path = "img/random/Boundary"
    # folder_path = 'media/images'
    folder_path = r'img\general_net\Boundary'
    # folder_path = r'img\general_net\Trajectories of surfnet graph edges'
    dur = 200
    animate(folder_path, dur, loop=0)