#docker rm bs_app_1 
export MY_IMAGE_NAME=sc_app_4
export MY_CONTAINER_NAME=sc_app_cnt_4

docker run -it --name $MY_CONTAINER_NAME --memory="1g" --memory-swap="1g" --cpus="1.0" -p 8004:8000 $MY_IMAGE_NAME 
