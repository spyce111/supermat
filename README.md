Setup Steps for successful container utilisation:
<be>
<pre>
Step 1: Install Docker
Step 2: build image 
        ex: docker build --no-cache -t super_mat_img1 .
Step 3: docker run on the bash mode 
        ex: docker run -it -p 8000:8000 -v /home/lenovo/Documents/NewProject/supermat/:/rest/supermat super_mat_img1 bash
         /home/lenovo/Documents/NewProject/supermat/ specifies the local path
         super_mat_img1 specifies the image name
Step 4: runserver command
        ex: python3 manage.py runserver 0:8000
</pre>
