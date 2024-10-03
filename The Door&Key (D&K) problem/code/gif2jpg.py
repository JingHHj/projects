from PIL import Image
import os

def split_gif(gif_path):
    
    # open .gif file
    gif = Image.open(gif_path)

    # put .gif file into different frame
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    return frames

def gif2jpg(name):
    
    gif_path = "./gif/" + name + ".gif"
    frames = split_gif(gif_path)

    
    for i, frame in enumerate(frames):
        directory = "./result/{}".format(name)
        if not os.path.exists(directory): # make sure there is a folder with the "name"
            os.makedirs(directory)
        frame.save("./result/{}/frame_{}.png".format(name,i), format="PNG")
    return None

def main():
    # gif2jpg()
    return None


if __name__ == "__main__":
    main()




