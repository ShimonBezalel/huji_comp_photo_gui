from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt


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
        response = HttpResponse('')
        response.status_code = 400
        return response
    return HttpResponse('')


def focus_image(request):
    try:
        value = request.GET.get('value')

    # load images to directory
    except:
        response = HttpResponse('')
        response.status_code = 400
        return response
    return HttpResponse('')


