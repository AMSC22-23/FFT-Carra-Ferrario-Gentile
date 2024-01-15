#!/bin/bash

#echo "[INFO] checking if mpiuser exists."
if ! sudo grep "mpiuser" /etc/passwd ;then
	echo "[INFO] mpiuser not found."
	sudo useradd -m mpiuser
	echo "[INFO] created new user mpiuser"
fi

sudo su mpiuser -c "mkdir -p shared_storage"
echo "[INFO] mounting shared_storage ($1) in /home/mpiuser/shared/storage"
sudo mount -t nfs $1:/home/mpiuser/shared_storage /home/mpiuser/shared_storage
echo "[INFO] worker configuration complete."
