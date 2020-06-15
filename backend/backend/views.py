import pprint

from django.http import HttpResponse, FileResponse, JsonResponse
from django.core.files.uploadedfile import TemporaryUploadedFile

from django.views.decorators.csrf import csrf_exempt
import os
import numpy as np
from matplotlib import pyplot as plt
import zipfile
import shutil
from backend import settings
from backend.settings import BASE_DIR
import random

from alg.image_api import Gui as Cache, open_series

# cache = None
# example = "EmekRefaim"
# suffix = 'hillel11_long_sdepth_ax18'
# p = os.path.join("sample_data", example)
#
# test_cache.setup(series_path=p, suffix=suffix, extension="jpg", zero_index=True, height=500, width=900)
#
# example = "apples"
# suffix = 'APPLE'
# p = os.path.join("sample_data", example)
#
# cache.setup(series_path=p, suffix=suffix, extension="jpg", height=500, width=900)
cache = None


def test(request):
    return HttpResponse("<html><body>Reached test!</body></html>")


@csrf_exempt
def upload_images(request):
    global cache
    file: TemporaryUploadedFile = request.FILES["images"]
    try:
        if not file or 'zip' not in file.content_type:
            raise ValueError("Must supply valid zip file.")
        name = file.name.replace(".zip", "").replace(" ", "_").strip()
        shutil.rmtree(settings.IMAGES_DIR)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(settings.IMAGES_DIR)
        path = os.path.join(settings.IMAGES_DIR, os.listdir(settings.IMAGES_DIR)[0])
        # Zip provided may contain files only, no dirs
        if not os.path.isdir(path):
            path = settings.IMAGES_DIR
        file_list = os.listdir(path)
        extension = os.path.splitext(file_list[0])[1].replace(".", "")
        if extension.lower() not in ['jpg', 'jpeg', 'png']:
            raise ValueError("Images' format must be jpeg or png, not {}".format(extension))
        image_only_list = list(filter(lambda p: p.endswith(extension), file_list))
        irrelevant_file_list = list(filter(lambda p: not p.endswith(extension), file_list))
        if len(image_only_list) > 999:
            raise ValueError("Up to 999 images supported, received {}".format(len(image_only_list)))
        image_only_list.sort()
        file_format = "{}{:03d}.{}" if len(image_only_list) > 100 else "{}{:02d}.{}"
        for count, filename in enumerate(image_only_list):
            os.rename(os.path.join(path, filename), os.path.join(path, file_format.format(name, count + 1, extension)))
        for filename in irrelevant_file_list:
            try:
                os.remove(os.path.join(path, filename))
            except:
                print("Could not remove {}".format(filename))
        cache = Cache(series_path=path, prefix=name, extension=extension)
        return JsonResponse({'rows': cache._rows,
                             'cols': cache._cols,
                             'channels': cache._channels,
                             'frames': cache._frames
                             })
    # load images to directory

    except Exception as e:
        print(e)
        response = HttpResponse(str(e))
        response.status_code = 400
        return response


@csrf_exempt
def slice(request):
    shift = float(request.GET.get('shift'))
    move = float(request.GET.get('move'))
    stereo = float(request.GET.get('stereo'))

    print("Calc Slice - move: {} stereo: {} shift: {}".format(move, stereo, shift))

    slice = cache._calc_slice(move, stereo, shift)
    raw = [int(i) for i in [slice[0][0], slice[0][1], slice[1][0], slice[1][1]]]

    return JsonResponse({'slice': raw})


def focus(request):
    res = cache.get_last_result()
    try:
        depth_input = request.GET.get('depth')
        center = int(request.GET.get('center'))
        radius = int(request.GET.get('radius'))
        depth = 1 - float(depth_input)
        # path = os.path.join(BASE_DIR, 'sample_data', 'apples', 'APPLE{:03d}.jpg'.format(int(value * 200)))
        assert 1 <= center <= cache._frames
        assert 1 <= radius <= cache._frames / 2

        res = cache.focus(depth, center, radius)
    except Exception as e:
        print(e)
    plt.imsave("tmp.jpeg", res)
    with open("tmp.jpeg", 'rb') as f:
        return HttpResponse(f.read(), content_type="image/jpeg")


def viewpoint(request):
    try:
        res: np.ndarray = cache.get_last_result()
        try:
            slice_raw = request.GET.get('slice')
            factor = request.GET.get('factor', None)
            if factor:
                factor = float(factor)
            if slice_raw not in [None, "", "()", "((),())", (), []]:
                print("Viewpoint - slice {}".format(slice_raw))
                inputs = [int(i) for i in slice_raw.split(",")]
                slice = (inputs[0], inputs[1]), (inputs[2], inputs[3])

                res = cache.viewpoint(slice=slice, factor=factor)
            else:
                shift = float(request.GET.get('shift'))
                move = float(request.GET.get('move'))
                stereo = float(request.GET.get('stereo'))
                print("Viewpoint - move: {} stereo: {} shift: {}".format(move, stereo, shift))

                res = cache.viewpoint(shift=shift, move=move, stereo=stereo, factor=factor)
        except Exception as e:
            print(e)

        plt.imsave("tmp.jpeg", res)
        with open("tmp.jpeg", 'rb') as f:
            return HttpResponse(f.read(), content_type="image/jpeg")
        # path = os.path.join(BASE_DIR, 'sample_data', 'apples', 'APPLE001.jpg')


    except Exception as e:
        print(e)
        response = HttpResponse('')
        response.status_code = 400
        return response


def motion(request):
    try:
        motion_vec = np.round(cache.get_motion_vec(), 3).tolist()
        payload = {'motion_vector': motion_vec}

        should_add_string = request.GET.get('add_string', default=False)
        if (should_add_string is not False) or should_add_string == 'true' or should_add_string == 'True' or \
                should_add_string == '1' or int(should_add_string) > 0:
            s = pprint.pformat(motion_vec, indent=4)
            payload['as_string'] = s

        return JsonResponse(payload)

    # load images to director
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response


def save(request):
    try:
        res = cache.get_last_result(resized=False)
        plt.imsave("tmp.jpeg", res)
        with open("tmp.jpeg", 'rb') as f:
            return HttpResponse(f.read(), content_type="image/jpeg")


    # load images to director
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response
