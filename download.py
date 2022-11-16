from simple_image_download import simple_image_download as simp
response = simp.simple_image_download
keywords = ["pig","hog","taylor swift","cow"]
for i in keywords:
    response().download(i,5000)