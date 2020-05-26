from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt
import os

from backend.settings import BASE_DIR


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
    try:
        value = request.GET.get('value')
        path = os.path.join(BASE_DIR, 'sample_data', 'apples', 'APPLE001.jpg')
        with open(path, "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpeg")

    # load images to director
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response


def viewpoint(request):
    try:
        shift_start = request.GET.get('shift_start')
        shift_end = request.GET.get('shift_end')
        move = request.GET.get('move')
        stereo = request.GET.get('stereo')
        path = os.path.join(BASE_DIR, 'sample_data', 'apples', 'APPLE001.jpg')
        with open(path, "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpeg")

    # load images to director
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response


def motion(request):
    try:
        return HttpResponse()

    # load images to director
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response
