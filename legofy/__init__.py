from __future__ import unicode_literals

from PIL import Image, ImageSequence
import sys
import os
import pandas as pd
import numpy as np

# Python 2 and 3 support
# TODO: Proper images2gif version that supports both Py 2 and Py 3 (mostly handling binary data)
if sys.version_info < (3,):
    import legofy.images2gif_py2 as images2gif
else:
    import legofy.images2gif_py3 as images2gif
from legofy import palettes

__size_brick_image__ = (None,None)
__vec_lego_brick_colors__ = []
__vec_lego_brick_ids__ = []
__df_lego_palette__ = pd.DataFrame()
__df_lego_parts_build__ = pd.DataFrame()

def __reset_internals__():
    global __vec_lego_brick_colors__, __vec_lego_brick_ids__, __df_lego_parts_build__
    
    __vec_lego_brick_colors__ = []
    __vec_lego_brick_ids__ = []
    __df_lego_parts_build__ = pd.DataFrame()

def get_brick_image_dims():
    return(__size_brick_image__)

def get_img_to_block_color_values():
    '''Retrns the set of colors for each block in the sequence of the image.'''
    return(__vec_lego_brick_colors__.copy())

def get_img_to_block_ids(col_r,col_g,col_b):
    '''Returns the first block ID identified by the color.'''
    if(__df_lego_palette__.shape[0]==0):
        get_palette_dataframe()
    tmp_block = __df_lego_palette__.loc[(__df_lego_palette__['col_r']==col_r) & (__df_lego_palette__['col_g']==col_g) & (__df_lego_palette__['col_b']==col_b),'lego_id'].tolist()
    rtn_val = None
    if(len(tmp_block)==0):
        print('ERROR: Block not found for color with r =',col_r,', g =',col_g,', b =',col_b)
    else:
        rtn_val = tmp_block[0]
    return(int(rtn_val))

def get_block_summary():
    '''Converts the colors into block IDs and provides a summary.'''
    global __vec_lego_brick_ids__, __df_lego_parts_build__
    for i,color_val in enumerate(__vec_lego_brick_colors__):
        try:
            __vec_lego_brick_ids__.append(get_img_to_block_ids(col_r=color_val[0],col_g=color_val[1],col_b=color_val[2]))
        except:
            print('ERROR: At index location',i,'for the image, with color value',[str(x) for x in color_val])
    arr_uniq,arr_cnts = np.unique(__vec_lego_brick_ids__,return_counts=True)
    __df_lego_parts_build__ = pd.DataFrame({'lego_id':arr_uniq,'count':arr_cnts})
    return(__df_lego_parts_build__)

def get_palette_dataframe():
    global __df_lego_palette__
    vec_col_r = []
    vec_col_g = []
    vec_col_b = []
    vec_col_type = []
    vec_col_id = []
    for i,p in enumerate(['solid','transparent','effects','mono']):
        vec_palette = palettes.LEGOS[p]
        for j,key in enumerate(vec_palette.keys()):
            tmp_r,tmp_g,tmp_b = vec_palette[key]
            vec_col_type.append(p)
            vec_col_id.append(key)
            vec_col_r.append(tmp_r)
            vec_col_g.append(tmp_g)
            vec_col_b.append(tmp_b)
    df_lego_colors = pd.DataFrame({'lego_id':vec_col_id,'type':vec_col_type,'col_r':vec_col_r,'col_g':vec_col_g,'col_b':vec_col_b})
    df_lego_colors['lego_id_num'] = pd.to_numeric(df_lego_colors['lego_id'])
    df_lego_colors.sort_values(by=['type','lego_id_num'],inplace=True)
    __df_lego_palette__ = df_lego_colors.copy()
    return(__df_lego_palette__)

def apply_color_overlay(image, color):
    '''Small function to apply an effect over an entire image'''
    overlay_red, overlay_green, overlay_blue = color
    channels = image.split()

    r = channels[0].point(lambda color: overlay_effect(color, overlay_red))
    g = channels[1].point(lambda color: overlay_effect(color, overlay_green))
    b = channels[2].point(lambda color: overlay_effect(color, overlay_blue))


    channels[0].paste(r)
    channels[1].paste(g)
    channels[2].paste(b)

    return Image.merge(image.mode, channels)

def overlay_effect(color, overlay):
    '''Actual overlay effect function'''
    if color < 33:
        return overlay - 100
    elif color > 233:
        return overlay + 100
    else:
        return overlay - 133 + color

def make_lego_image(thumbnail_image, brick_image):
    '''Create a lego version of an image from an image'''
    global __vec_lego_brick_colors__
    base_width, base_height = thumbnail_image.size
    brick_width, brick_height = brick_image.size

    rgb_image = thumbnail_image.convert('RGB')

    lego_image = Image.new("RGB", (base_width * brick_width,
                                   base_height * brick_height), "white")
    
    __vec_lego_brick_colors__ = []
    for brick_x in range(base_width):
        for brick_y in range(base_height):
            color = rgb_image.getpixel((brick_x, brick_y))
            __vec_lego_brick_colors__.append(color)
            lego_image.paste(apply_color_overlay(brick_image, color),
                             (brick_x * brick_width, brick_y * brick_height))
    return lego_image


def get_new_filename(file_path, ext_override=None):
    '''Returns the save destination file path'''
    folder, basename = os.path.split(file_path)
    base, extention = os.path.splitext(basename)
    if ext_override:
        extention = ext_override
    new_filename = os.path.join(folder, "{0}_lego{1}".format(base, extention))
    return new_filename


def get_new_size(base_image, brick_image, size=None):
    '''Returns a new size the first image should be so that the second one fits neatly in the longest axis'''
    global __size_brick_image__
    new_size = base_image.size
    if size:
        scale_x, scale_y = size, size
    else:
        scale_x, scale_y = brick_image.size

    if new_size[0] > scale_x or new_size[1] > scale_y:
        if new_size[0] < new_size[1]:
            scale = new_size[1] / scale_y
        else:
            scale = new_size[0] / scale_x

        new_size = (int(round(new_size[0] / scale)) or 1,
                    int(round(new_size[1] / scale)) or 1)
    __size_brick_image__ = new_size
    return new_size

def get_lego_palette(palette_mode):
    '''Gets the palette for the specified lego palette mode'''
    legos = palettes.legos()
    palette = legos[palette_mode]
    return palettes.extend_palette(palette)


def apply_thumbnail_effects(image, palette, dither):
    '''Apply effects on the reduced image before Legofying'''
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(palette)
    return image.im.convert("P",
                        Image.FLOYDSTEINBERG if dither else Image.NONE,
                        palette_image.im)

def legofy_gif(base_image, brick_image, output_path, size, palette_mode, dither):
    '''Alternative function that legofies animated gifs, makes use of images2gif - uses numpy!'''
    im = base_image

    # Read original image duration
    original_duration = im.info['duration']

    # Split image into single frames
    frames = [frame.copy() for frame in ImageSequence.Iterator(im)]

    # Create container for converted images
    frames_converted = []

    print("Number of frames to convert: " + str(len(frames)))

    # Iterate through single frames
    for i, frame in enumerate(frames, 1):
        print("Converting frame number " + str(i))

        new_size = get_new_size(frame, brick_image, size)
        frame = frame.resize(new_size, Image.ANTIALIAS)
        if palette_mode:
            palette = get_lego_palette(palette_mode)
            frame = apply_thumbnail_effects(frame, palette, dither)
        new_frame = make_lego_image(frame, brick_image)
        frames_converted.append(new_frame)

    # Make use of images to gif function
    images2gif.writeGif(output_path, frames_converted, duration=original_duration/1000.0, dither=0, subRectangles=False)

def legofy_image(base_image, brick_image, output_path, size, palette_mode, dither):
    '''Legofy an image'''
    new_size = get_new_size(base_image, brick_image, size)
    base_image = base_image.resize(new_size, Image.ANTIALIAS)

    if palette_mode:
        palette = get_lego_palette(palette_mode)
        base_image = apply_thumbnail_effects(base_image, palette, dither)
    make_lego_image(base_image, brick_image).save(output_path)


def main(image_path, output_path=None, size=None,
         palette_mode=None, dither=False):
    '''Legofy image or gif with brick_path mask'''
    
    __reset_internals__()
    
    image_path = os.path.realpath(image_path)
    if not os.path.isfile(image_path):
        print('Image file "{0}" was not found.'.format(image_path))
        sys.exit(1)

    brick_path = os.path.join(os.path.dirname(__file__), "assets",
                              "bricks", "1x1.png")

    if not os.path.isfile(brick_path):
        print('Brick asset "{0}" was not found.'.format(brick_path))
        sys.exit(1)

    base_image = Image.open(image_path)
    brick_image = Image.open(brick_path)

    if palette_mode:
        print ("LEGO Palette {0} selected...".format(palette_mode.title()))
    elif dither:
        palette_mode = 'all'

    if image_path.lower().endswith(".gif") and base_image.is_animated:
        if output_path is None:
            output_path = get_new_filename(image_path)
        print("Animated gif detected, will now legofy to {0}".format(output_path))
        legofy_gif(base_image, brick_image, output_path, size, palette_mode, dither)
    else:
        if output_path is None:
            output_path = get_new_filename(image_path, '.png')
        print("Static image detected, will now legofy to {0}".format(output_path))
        legofy_image(base_image, brick_image, output_path, size, palette_mode, dither)

    base_image.close()
    brick_image.close()
    print("Finished!")