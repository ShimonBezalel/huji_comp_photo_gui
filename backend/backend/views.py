from django.http import HttpResponse


def test(request):
    return HttpResponse("<html><body>Reached test!</body></html>")
