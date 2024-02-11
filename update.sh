sudo docker stop emotion
sudo docker rm emotion
sudo docker system prune -a
sudo docker build -t emotion .
sudo docker run -d -p 5000:5000 --name emotion emotion
sudo docker logs -f -t emotion