import pprint

from django.http import HttpResponse, FileResponse, JsonResponse

from django.views.decorators.csrf import csrf_exempt
import os
import numpy as np
from matplotlib import pyplot as plt


from backend.settings import BASE_DIR
import random

from alg.image_api import Gui as Cache

test_cache = Cache()

example = "EmekRefaim"
suffix = 'hillel11_long_sdepth_ax18'
p = os.path.join("sample_data", example)

test_cache.setup(series_path=p, suffix=suffix, extension="jpg", zero_index=True, height=500, width=900)
#
# example = "apples"
# suffix = 'APPLE'
# p = os.path.join("sample_data", example)
#
# test_cache.setup(series_path=p, suffix=suffix, extension="jpg", height=500, width=900)



def test(request):
    return HttpResponse("<html><body>Reached test!</body></html>")



@csrf_exempt
def upload_images(request):
    file = request.FILES["images"]

    try:
        if not file:
            raise ValueError
    # load images to directory

    except:
        response = HttpResponse(request.request.FILES)
        response.status_code = 400
        return response
    return HttpResponse('')


#
# def focus_image(request):
#     try:
#         value = request.GET.get('value')
#
#         try:
#             with open(valid_image, "rb") as f:
#                 return HttpResponse(f.read(), content_type="image/jpeg")
#         except IOError:
#             red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
#             response = HttpResponse(content_type="image/jpeg")
#             red.save(response, "JPEG")
#             return response
#
#     # load images to directory
#     except:
#         response = HttpResponse('')
#         response.status_code = 400
#         return response
#     return HttpResponse('')


def focus(request):
    res = test_cache.get_last_result()
    try:
        depth = request.GET.get('depth')
        depth = float(depth)
        # path = os.path.join(BASE_DIR, 'sample_data', 'apples', 'APPLE{:03d}.jpg'.format(int(value * 200)))
        shift_factor = test_cache.get_motion_avg() * depth
        res = test_cache.focus(shift_factor)
    except Exception as e:
        print(e)
    plt.imsave("tmp.jpeg", res)
    with open("tmp.jpeg", 'rb') as f:
        return HttpResponse(f.read(), content_type="image/jpeg")



def viewpoint(request):
    try:
        res: np.ndarray = test_cache.get_last_result()
        try:
            slice = request.GET.get('slice')
            if slice not in [None, "", "()", "((),())", (), []]:
                print("Viewpoint - slice {}".format(slice))

                res = test_cache.viewpoint(slice=slice)
            else:
                shift = float(request.GET.get('shift'))
                move = float(request.GET.get('move'))
                stereo = float(request.GET.get('stereo'))
                print("Viewpoint - move: {} stereo: {} shift: {}".format(move, stereo, shift))

                res = test_cache.viewpoint(shift=shift, move=move, stereo=stereo)
        except Exception as e:
            print(e)

        plt.imsave("tmp.jpeg", res)
        with open("tmp.jpeg", 'rb') as f:
            return HttpResponse(f.read(), content_type="image/jpeg")
        # path = os.path.join(BASE_DIR, 'sample_data', 'apples', 'APPLE001.jpg')


    # load images to director
    except Exception as e:
        print(e)
        response = HttpResponse('')
        response.status_code = 400
        return response


def motion(request):
    try:
        motion_vec = np.round(test_cache.get_motion_vec(), 3).tolist()
        s = pprint.pformat(motion_vec, indent=4)
        return JsonResponse({'motion_vector': motion_vec, 'as_string': s})

    # load images to director
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response
